from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

from botocore.exceptions import NoCredentialsError, ClientError
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
from openai import OpenAI
from io import BytesIO

import requests
import base64
import boto3
import json
import os

app = Flask(__name__)

register_heif_opener()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)


# 원격 이미지 URL을 Base64로 인코딩하는 함수
def encode_remote_image_to_base64(url: str) -> str | None:
  try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    image_data = BytesIO(response.content)
    image = Image.open(image_data).convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_str
  except requests.exceptions.RequestException as e:
    print(f"오류: URL 요청 실패 - {url}: {e}")
    return None
  except UnidentifiedImageError as e:
    print(f"오류: 이미지 식별 실패 - {url}: {e}")
    if 'response' in locals() and response.content:
      print(f"다운로드된 콘텐츠 시작 부분 (디버깅용): {response.content[:100]}...")
    return None
  except Exception as e:
    print(f"오류: 원격 이미지 인코딩 중 예상치 못한 오류 발생 - {url}: {e}")
    return None


# 로컬 이미지 파일을 Base64로 인코딩하는 함수
def encode_local_image_to_base64(image_path: str) -> str | None:
  try:
    with open(image_path, "rb") as image_file:
      image = Image.open(image_file).convert("RGB")

      buffer = BytesIO()
      image.save(buffer, format="JPEG")
      base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
      return base64_str
  except FileNotFoundError:
    print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {image_path}")
    return None
  except UnidentifiedImageError as e:
    print(f"오류: 이미지 식별 실패 - {image_path}: {e}")
    return None
  except Exception as e:
    print(f"오류: 로컬 이미지 인코딩 중 예상치 못한 오류 발생 - {image_path}: {e}")
    return None


# OpenAI GPT-4o를 사용하여 이미지 분석을 수행하는 함수
def analyze_images_with_gpt4o(base64_data_urls: list[str],
    prompt_text: str) -> str | None:
  try:
    print(f"GPT-4o를 사용하여 {len(base64_data_urls)}개 이미지와 텍스트 프롬프트 분석 요청 중...")

    content_messages = []
    content_messages.append({"type": "text", "text": prompt_text})
    for url in base64_data_urls:
      content_messages.append({"type": "image_url", "image_url": {"url": url}})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
          {
            "role": "user",
            "content": content_messages,
          }
        ],
        max_tokens=2000,
        response_format={"type": "json_object"}
    )

    analysis_content = response.choices[0].message.content
    return analysis_content
  except Exception as e:
    print(f"오류: OpenAI API 호출 중 오류 발생 - {e}")
    return None


# S3 URL을 파싱하여 버킷 이름과 폴더 경로를 추출하는 함수
def parse_s3_url(s3_url: str) -> tuple[str, str, str] | None:
  if not s3_url.startswith("s3://"):
    return None
  parts = s3_url[len("s3://"):].split('/', 1)
  bucket_name = parts[0]
  prefix = parts[1] if len(parts) > 1 else ''
  return bucket_name, prefix, None  # 마지막 None은 key_name이므로 사용하지 않음


# S3에서 이미지를 다운로드하여 BytesIO 객체로 반환하는 함수
def download_image_from_s3(bucket_name: str, key: str) -> BytesIO | None:
  try:
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    image_data = BytesIO(response['Body'].read())
    return image_data
  except NoCredentialsError:
    print("오류: AWS 자격 증명이 설정되지 않았습니다.")
    return None
  except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == 'NoSuchKey':
      print(f"오류: S3 키를 찾을 수 없습니다 - {key}")
    elif error_code == 'AccessDenied':
      print(f"오류: S3 접근 거부 - {key}")
    else:
      print(f"오류: S3 다운로드 중 예상치 못한 오류 발생 - {key}: {e}")
    return None
  except Exception as e:
    print(f"오류: S3 이미지 다운로드 중 일반 오류 발생 - {key}: {e}")
    return None


# S3 폴더 내 이미지를 분석하는 Flask 엔드포인트
@app.route('/analyze_s3_images', methods=['POST'])
def analyze_s3_images():
  """
  Flask 엔드포인트: memberId와 retripId를 받아 S3 폴더의 이미지를 분석
  """
  try:
    # JSON 요청 데이터 파싱 시도
    data = request.get_json(force=True)
    if not data:
      return jsonify({"error": "요청 본문이 JSON 형식이 아니거나 비어 있습니다."}), 400

    print(f"수신된 데이터: {data}")

    # Spring Boot에서 직접 객체를 보내는 경우
    member_id = data.get('memberId')
    retrip_id = data.get('retripId')
    main_location_lat = data.get('mainLocationLat')
    main_location_lng = data.get('mainLocationLng')

    # 객체가 다른 필드에 감싸져 있을 경우 (request, body 등의 필드 확인)
    if not member_id and 'request' in data:
      member_id = data['request'].get('memberId')
      retrip_id = data['request'].get('retripId')
      main_location_lat = data['request'].get('mainLocationLat')
      main_location_lng = data['request'].get('mainLocationLng')
    elif not member_id and 'body' in data:
      member_id = data['body'].get('memberId')
      retrip_id = data['body'].get('retripId')
      main_location_lat = data['body'].get('mainLocationLat')
      main_location_lng = data['body'].get('mainLocationLng')

    if not member_id or not retrip_id:
      return jsonify({"error": "memberId와 retripId가 필요합니다."}), 400

    print(f"memberId = {member_id} / retripId = {retrip_id}")
    print(f"main_location_lat = {main_location_lat} / main_location_lng = {main_location_lng}")
    print(f"bucket_name = {os.getenv('AWS_BUCKET_NAME')}")
    # S3 폴더 경로 생성
    s3_folder_prefix = f"{member_id}/{retrip_id}/"
    bucket_name = os.getenv('AWS_BUCKET_NAME')

    if not bucket_name:
      return jsonify({"error": "AWS_BUCKET_NAME 환경 변수가 설정되지 않았습니다."}), 500

    print(
        f"S3 버킷: {bucket_name}, 폴더/접두사: {s3_folder_prefix} 에서 이미지 목록을 가져오는 중...")

    try:
      # S3 버킷 내의 파일 목록 가져오기
      response = s3_client.list_objects_v2(Bucket=bucket_name,
                                           Prefix=s3_folder_prefix)
      s3_objects = response.get('Contents', [])

      if not s3_objects:
        return jsonify({"message": f"'{s3_folder_prefix}' 폴더에 이미지가 없습니다."}), 200

      processed_image_urls = []
      failed_images_info = []

      allowed_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.heic')

      print("--- S3 이미지 다운로드 및 Base64 인코딩 시작 ---")
      for s3_object in s3_objects:
        key = s3_object['Key']
        if key.endswith('/'):
          continue

        file_ext = os.path.splitext(key)[1].lower()
        if file_ext not in allowed_extensions:
          print(f"건너뛰기: 지원하지 않는 파일 형식 - {key} ({file_ext})")
          failed_images_info.append(
              {"id": key, "reason": f"지원하지 않는 파일 형식: {file_ext}"})
          continue

        print(f"다운로드 중: {key}")
        image_data_stream = download_image_from_s3(bucket_name, key)

        if image_data_stream:
          try:
            image = Image.open(image_data_stream).convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            processed_image_urls.append(f"data:image/jpeg;base64,{base64_str}")
            print(f"성공적으로 인코딩 완료: {key}")
          except UnidentifiedImageError as e:
            print(f"오류: S3 이미지 식별 실패 - {key}: {e}")
            failed_images_info.append({"id": key, "reason": f"이미지 식별 실패: {e}"})
          except Exception as e:
            print(f"오류: S3 이미지 처리 중 예상치 못한 오류 발생 - {key}: {e}")
            failed_images_info.append(
                {"id": key, "reason": f"인코딩 중 예상치 못한 오류: {e}"})
        else:
          failed_images_info.append({"id": key, "reason": "S3 다운로드 실패"})

      if not processed_image_urls:
        return jsonify({
          "message": f"'{s3_folder_prefix}' 폴더에서 분석할 이미지를 찾을 수 없거나 모두 처리 실패했습니다.",
          "failed_images_info": failed_images_info
        }), 200


      travel_analysis_prompt = f"""
        당신은 여행 사진 분석 및 추천 전문가입니다. 여행 사진들과 위치 데이터({main_location_lat}, {main_location_lng})를 바탕으로 사용자의 여행 리캡 정보를 생성해주세요.
        위치데이터는 여행사진들 중 가장 사진을 많이 찍은 곳의 좌표입니다.
        다음 형식의 JSON으로 응답해 주세요:

        {{
          "user": {{
            "countryCode": "방문한 국가의 ISO 코드 (예: KR, JP, US 등)",
            "mbti": "사용자의 여행 스타일을 반영한 MBTI (# 포함)"
          }},
          "tripSummary": {{
            "summaryLine": "여행 분위기와 특성을 반영한 한 줄 요약 (20자 이내, 언더바_허용)",
            "keywords": ["여행을 표현하는 키워드 3개 (# 포함)", "..."],
            "hashtag": "#단일단어해시태그"
          }},
          "photoStats": {{
            "favoriteSubjects": ["피사체를 나타내는 이모지 3개만 (예: 🏛️, 🌊, 👨‍👩‍👧‍👦)", "..."],
            "favoritePhotoSpot": "가장 많이 촬영된 장소명 (한글)"
          }},
          "recommendations": [
            {{
              "emoji": "장소 특성을 나타내는 이모지 1개",
              "place": "추천 장소명",
              "description": "장소에 대한 간략한 설명 (10자 이내)"
            }},
            "..." // 총 5개 추천
          ]
        }}

        분석 시 다음을 고려해주세요:
        1. 사진의 주요 피사체와 배경을 분석하여 여행의 전반적인 분위기를 파악하세요.
        2. 방문 국가는 제공된 위도/경도를 기반으로 정확히 판단하세요.
        3. 추천 장소는 방문한 주요 장소와 유사하거나 근처에 있는 인기 관광지여야 합니다.
        4. 키워드는 여행의 감정, 분위기, 특성을 반영해야 합니다.
        5. MBTI는 사진의 구도, 피사체 선택, 여행 스타일을 바탕으로 추정하세요.
        6. 가장 많이 촬영된 장소명은 위치데이터에 기반해서 주요장소를 추정해서 반환해주세요.
        7. 반환하는 JSON 구조를 정확히 지켜주세요 - 필드명과 데이터 형식이 중요합니다.
        """

      print("\n--- 전체 여행 이미지 분석 요청 시작 ---")
      combined_analysis_json_str = analyze_images_with_gpt4o(
          processed_image_urls, travel_analysis_prompt)

      if combined_analysis_json_str:
        try:
          combined_analysis_data = json.loads(combined_analysis_json_str)
          return jsonify({
            "travel_image_analysis": combined_analysis_data,
            "failed_images_info": failed_images_info
          }), 200
        except json.JSONDecodeError as e:
          print(f"오류: GPT-4o 통합 응답 파싱 실패 (유효하지 않은 JSON): {e}")
          print(f"모델 응답 내용: {combined_analysis_json_str[:500]}...")
          return jsonify({
            "error": "GPT-4o 응답 파싱 실패",
            "raw_openai_response": combined_analysis_json_str,
            "failed_images_info": failed_images_info
          }), 500
      else:
        return jsonify({
          "error": "통합 이미지 분석 실패 (API 호출 오류 또는 응답 없음)",
          "failed_images_info": failed_images_info
        }), 500

    except NoCredentialsError:
      return jsonify(
          {"error": "AWS 자격 증명이 설정되지 않았습니다. 환경 변수 또는 IAM 역할을 확인해주세요."}), 500
    except ClientError as e:
      return jsonify({
        "error": f"S3 클라이언트 오류: {e.response['Error']['Code']} - {e.response['Error']['Message']}"}), 500
    except Exception as e:
      print(f"예상치 못한 서버 오류: {e}")
      return jsonify({"error": f"서버 내부 오류 발생: {e}"}), 500

  except json.JSONDecodeError as e:
    return jsonify({"error": f"JSON 파싱 오류: {e}"}), 400
  except Exception as e:
    print(f"예상치 못한 서버 오류: {e}")
    return jsonify({"error": f"서버 내부 오류 발생: {e}"}), 500


if __name__ == '__main__':
  # Flask 애플리케이션 실행
  app.run(debug=True, host='0.0.0.0',
          port=5000)
