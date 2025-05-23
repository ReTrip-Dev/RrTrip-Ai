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
  data = request.get_json()
  s3_folder_url = data.get('s3_folder_url')

  if not s3_folder_url:
    return jsonify({"error": "s3_folder_url이 제공되지 않았습니다."}), 400

  parsed_url = parse_s3_url(s3_folder_url)
  if not parsed_url:
    return jsonify({"error": "유효하지 않은 S3 URL 형식입니다."}), 400

  bucket_name, folder_prefix, _ = parsed_url
  print(f"S3 버킷: {bucket_name}, 폴더/접두사: {folder_prefix} 에서 이미지 목록을 가져오는 중...")

  try:
    # S3 버킷 내의 파일 목록 가져오기
    response = s3_client.list_objects_v2(Bucket=bucket_name,
                                         Prefix=folder_prefix)
    s3_objects = response.get('Contents', [])

    if not s3_objects:
      return jsonify({"message": f"'{s3_folder_url}' 폴더에 이미지가 없습니다."}), 200

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
          # HEIC 파일도 Pillow-HEIF 덕분에 여기서 JPEG로 변환됩니다.
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
        "message": f"'{s3_folder_url}' 폴더에서 분석할 이미지를 찾을 수 없거나 모두 처리 실패했습니다.",
        "failed_images_info": failed_images_info
      }), 200

    travel_analysis_prompt = """
        당신은 여행 사진 분석 전문가입니다. 제공된 모든 여행 이미지들을 종합적으로 분석하여 다음 세 가지 정보를 JSON 형식으로 제공해 주세요.
        이번 여행의 중심지의 위도, 경도 값은 다음과 같아. (35.102245, 129.030598)

        1.  **overall_mood**: 이번 여행 사진의 전반적인 분위기를 한 문장으로 요약해 주세요.
        2.  **top5_subjects**: 이번 여행 사진에서 가장 자주 등장하는 피사체 (인물, 동물, 특정 건물, 자연물 등) 상위 5개를 빈도 순으로 나열하고 각 항목의 대략적인 개수를 포함해 주세요.
        3.  **photo_category_ratio**: 인물 사진, 풍경 사진, 음식 사진, 기타 사진의 대략적인 비율을 퍼센테이지로 제공해 주세요. (예: "인물: 30%, 풍경: 50%, 음식: 10%, 기타: 10%") 만약 특정 카테고리의 사진이 없거나 적다면 0%로 표시할 수 있습니다.
        4.  **top_visit_place**: 이번 여행에서 가장 많이 방문한 위도, 경도 값에 대해서 가장 많이 등장한 장소를 제공해 주세요. (예: {"latitude": 35.102245, "longitude": 129.030598, "place_name": "부산 해운대"})
        5.  **mbti**: 이번 여행의 전반적인 분위기와 데이터를 토대로 사용자의 MBTI 유형을 추정해 주세요. (예: "ISFJ", "ENTP" 등)

        다음 형식의 JSON으로 응답해 주세요:
        {
          "travel_analysis": {
            "overall_mood": "이번 여행 사진의 전반적인 분위기는...",
            "top5_subjects": [
              {"subject": "피사체1", "count": 10},
              {"subject": "피사체2", "count": 8},
              {"subject": "피사체3", "count": 5},
              {"subject": "피사체4", "count": 3},
              {"subject": "피사체5", "count": 2}
            ],
            "photo_category_ratio": {
              "people": "30%",
              "landscape": "50%",
              "food": "10%",
              "other": "10%"
            },
            "top_visit_place": {
              "latitude": 35.102245,
              "longitude": 129.030598,
              "place_name": "부산 해운대"
            },
            "mbti": "ISFJ"
          }
        }
        """

    print("\n--- 전체 여행 이미지 분석 요청 시작 ---")
    combined_analysis_json_str = analyze_images_with_gpt4o(processed_image_urls,
                                                           travel_analysis_prompt)

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


if __name__ == '__main__':
  # Flask 애플리케이션 실행
  app.run(debug=True, host='0.0.0.0',
          port=5000)
