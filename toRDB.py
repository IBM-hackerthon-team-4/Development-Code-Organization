import re
import requests
import os
import json
import time
import mysql.connector
from mysql.connector import Error

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

def create_llm(api_key, api_url, project_id):
    print("Creating Watsonx AI model object...")
    params = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY.value,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 1000,
        GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
    }
    creds = Credentials(url=api_url, api_key=api_key)
    model = ModelInference(
        model_id="mistralai/mistral-large",
        params=params,
        credentials=creds,
        project_id=project_id
    )
    print("Watsonx AI model object created successfully.")
    return model

def get_image_urls():
    keyword = "경진대회 공모전"
    print(f"Searching for image URLs with keyword: {keyword}")
    google_api_key = "AIzaSyAakKrTq1NikGxyEfX678VEytg8x25BEho"
    google_cx = "461254d9dc19a4cf0"
    url = f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={google_cx}&q={keyword}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Error fetching Google results: {e}")
        return []
    out = []
    items = data.get("items", [])
    print(f"Found {len(items)} items from Google search.")
    for item in items:
        pagemap = item.get("pagemap", {})
        cse_images = pagemap.get("cse_image", [])
        for img in cse_images:
            src = img.get("src")
            if src:
                out.append(src)
    print(f"Extracted {len(out)} image URLs.")
    return out

def check_and_reconnect(conn):
    try:
        if conn is None or not conn.is_connected():
            print("DB connection is down, reconnecting...")
            conn.reconnect(attempts=3, delay=5)
        return True
    except Error as e:
        print(f"Failed to reconnect: {e}")
        return False

def call_clova_ocr(image_url):
    print(f"Calling Clova OCR for URL: {image_url}")
    clova_url = "https://323nxlagj3.apigw.ntruss.com/custom/v1/38439/5fdf0aa9a2101b672d6d33aaea95fe59183b4434379ea592717f44ad9c0a2ad9/general"
    secret_key = "SkpKSnpRdHVKS1lDU3ZNTk1IUWRQdmVSV05yUnRIWGc="
    headers = {
        "Content-Type": "application/json",
        "X-OCR-SECRET": secret_key
    }
    request_body = {
        "version": "V2",
        "requestId": "sample_id",
        "timestamp": 0,
        "images": [
            {
                "name": "ocr_image",
                "format": "jpg",
                "url": image_url
            }
        ]
    }
    try:
        r = requests.post(clova_url, headers=headers, data=json.dumps(request_body), timeout=30)
        if r.status_code != 200:
            print(f"Clova OCR request failed with status code {r.status_code}")
            return ""
        result = r.json()
        images_info = result.get("images", [])
        if not images_info:
            print("No OCR info found in response.")
            return ""
        fields = images_info[0].get("fields", [])
        if not fields:
            print("No fields in OCR result.")
        extracted = "\n".join(x.get("inferText", "") for x in fields if x.get("inferText"))
        print("OCR extraction complete.")
        return extracted
    except Exception as e:
        print(f"Error calling Clova OCR: {e}")
        return ""

def extract_json_from_response(response_text):
    """
    Watsonx AI 응답에서 JSON 블록( ```json ... ``` )만 골라 파싱.
    첫 번째로 찾은 JSON 블록을 우선 파싱하며,
    찾지 못하면 통째로 다시 시도.
    """
    # 정규표현식: ```json 과 ``` 사이에 있는 { } 구간 (DOTALL로 개행 포함)
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        json_part = match.group(1).strip()
        try:
            return json.loads(json_part)
        except json.JSONDecodeError:
            pass
    # 만약 ```json 블록이 없다면, 그냥 전체를 JSON이라 가정하고 시도
    # (Watsonx AI가 JSON만 깔끔히 반환하는 경우)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {}

def process_image(image_url, watson_model):
    print(f"Processing URL for Watsonx AI: {image_url}")
    ocr_text = call_clova_ocr(image_url)
    if not ocr_text.strip():
        print("No OCR text extracted, skipping Watsonx AI processing.")
        return {}
    prompt = (
        "첨부하는 이미지는 대한민국에서 개최되는 공모전 또는 대회의 홍보 포스터야. "
        "해당 이미지에서 응시 대상자, 기간, 분야, 주최사, 시상내역의 내용을 추출하여 다음 json 형식으로 반환해."
        "{'제목': '포스터 내 제목을 그대로 사용', '응시 대상자': '제한없음, 일반인, 대학생, 청소년, 어린이, 기타 중 택 1', "
        "'기간': '전체, 일주일 이내, 한 달 이내, 3개월 이내, 6개월 이내, 6개월 이상 중 택 1', "
        "'분야': '기획/아이디어, 광고/마케팅, 논문/리포트/ 영상/UCC/사진, 디자인캐릭터웹툰, 웹/모바일/IT, 게임/소프트웨어, 과학/공학, "
        "문학/글/시나리오, 건축/건설/인테리어, 네이밍/슬로건, 예체능/미술/음악, 대외활동/서포터즈, 봉사활동, 취업/창업, 해외, 기타 중 택 1', "
        "'주최사': '정부/공공기관, 공기업, 대기업, 신문/방송/언론, 외국계기업, 중견/중소/벤처기업, 비영리/협회/재단, 해외, 기타 중 택 1', "
        "'시상내역': '100만원 이내, 100~500만원, 500~1000만원, 1000만원 이상, 취업특전, 입사시가산점, 인턴채용, 정직원채용 중 택 1'}"
        "이 때, 각 항목 중 가장 가까운 분류를 선택하도록 하고, 전혀 해당내용이 없을 경우에만 기타를 반환해.\n"
        + ocr_text
    )
    print("Sending prompt to Watsonx AI...")
    try:
        response = watson_model.generate(prompt=prompt)
        raw_text = response["results"][0]["generated_text"]
        print("Watsonx AI raw response:")
        print(raw_text)
        parsed_json = extract_json_from_response(raw_text.strip())
        print(f"Parsed JSON from Watsonx AI: {parsed_json}")
        return parsed_json
    except Exception as e:
        print(f"Error calling Watsonx AI: {e}")
        return {}

def insert_competition_data(cursor, conn, data):
    print(f"Inserting data into database: {data}")
    sql = "INSERT INTO competition (title, target, period, category, org, award, url) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    values = (
        data.get('제목'),
        data.get('응시 대상자'),
        data.get('기간'),
        data.get('분야'),
        data.get('주최사'),
        data.get('시상내역'),
        None
    )
    cursor.execute(sql, values)
    conn.commit()
    print("Insert successful.")

def run_process():
    image_urls = get_image_urls()
    if not image_urls:
        print("추출된 이미지 링크가 없습니다.")
        return
    print(f"Found {len(image_urls)} image URLs in total.")
    db_config = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': 'tkwh8304*',
        'database': 'competition',
        'connection_timeout': 180
    }
    api_key = "YGbzIUg6KpNsg7WPz3YpcrhFiqS5AElruCqpEmhPGkN9"
    api_url = "https://us-south.ml.cloud.ibm.com"
    project_id = "5c7be186-19b4-43b1-8606-82ea24e1e840"
    watson_model = create_llm(api_key, api_url, project_id)
    conn = None
    cursor = None
    try:
        print("Connecting to MySQL database...")
        conn = mysql.connector.connect(**db_config)
        print("Database connected.")
        cursor = conn.cursor()
        for idx, url in enumerate(image_urls, start=1):
            print(f"\nProcessing image #{idx}: {url}")
            if not check_and_reconnect(conn):
                print("Database reconnection failed, stopping process.")
                break
            try:
                data = process_image(url, watson_model)
                if data:
                    insert_competition_data(cursor, conn, data)
                else:
                    print("No data returned from Watsonx AI, skipping DB insert.")
                time.sleep(1)
            except Exception as e:
                print(f"Error processing {url}: {e}")
        if cursor:
            print("Closing cursor...")
            cursor.close()
        if conn and conn.is_connected():
            print("Closing DB connection...")
            conn.close()
    except Error as err:
        print(f"Database error: {err}")
    finally:
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if conn and conn.is_connected():
            try:
                conn.close()
                print("DB connection closed.")
            except:
                pass

def main():
    while True:
        print("\n===== Start of scheduled process =====")
        run_process()
        print("===== End of scheduled process. Sleeping for 30 minutes... =====\n")
        time.sleep(1800)

if __name__ == "__main__":
    main()
