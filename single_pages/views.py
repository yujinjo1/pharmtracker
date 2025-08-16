# views.py
import base64
import difflib
from bs4 import BeautifulSoup
import urllib.parse
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse
import os
import uuid
import json
import time
import platform
import numpy as np
import cv2
import requests
from PIL import ImageFont, ImageDraw, Image
import openai  # Assuming you're using OpenAI's GPT-3/4 as the chatbot backend
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from datetime import date
from xml.etree import ElementTree
import urllib.parse
from urllib.parse import unquote
from django.core.cache import cache
from statistics import mean
import re
from io import BytesIO

from django.views.decorators.http import require_http_methods
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add your OpenAI API key here
openai.api_key = os.environ.get("OPENAI_API_KEY")

api_url = os.environ.get("NAVER_OCR_API_URL")
secret_key = os.environ.get("NAVER_OCR_SECRET_KEY")

@login_required
def home_view(request):
   return render(request, 'home.html')

def splash_view(request):
    return render(request, 'splash.html')

@login_required
def mypage_view(request):
    user = request.user

    if request.method == 'POST':
        if 'delete_medication' in request.POST:
            # 삭제할 약품의 인덱스를 가져옵니다.
            index = int(request.POST.get('delete_medication'))
            medications = request.session.get('medications', [])

            if 0 <= index < len(medications):
                deleted_med = medications.pop(index)
                request.session['medications'] = medications
                request.session.modified = True
                messages.success(request, f"{deleted_med['item_name']}이(가) 내 약품 목록에서 삭제되었습니다.")
            else:
                messages.error(request, "유효하지 않은 약품입니다.")

            return redirect('mypage')

        elif 'gender' in request.POST and 'age' in request.POST:
            # 프로필 업데이트 처리
            gender = request.POST.get('gender')
            age = request.POST.get('age')
            user.first_name = gender
            user.last_name = age
            user.save()
            messages.success(request, '프로필이 업데이트되었습니다.')

        elif 'blood_pressure' in request.POST and 'blood_sugar' in request.POST and 'weight' in request.POST:
            # 건강 기록 저장 처리
            blood_pressure = request.POST.get('blood_pressure')
            blood_sugar = request.POST.get('blood_sugar')
            weight = request.POST.get('weight')
            today = date.today().strftime('%Y-%m-%d')

            health_record = {
                'blood_pressure': blood_pressure,
                'blood_sugar': blood_sugar,
                'weight': weight,
            }

            # 세션 데이터 초기화 시 딕셔너리로 설정
            if 'health_records' not in request.session:
                request.session['health_records'] = {}

            if not isinstance(request.session['health_records'], dict):
                request.session['health_records'] = {}

            if today not in request.session['health_records']:
                request.session['health_records'][today] = []

            request.session['health_records'][today].append(health_record)
            request.session.modified = True

            messages.success(request, '오늘의 건강 기록이 저장되었습니다.')

        return redirect('mypage')

    age_range = range(1, 101)
    health_records = request.session.get('health_records', {})
    medications = request.session.get('medications', [])
    context = {
        'age_range': age_range,
        'user': user,
        'health_records': health_records,
        'medications': medications,
    }
    return render(request, 'mypage.html', context)
def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if User.objects.filter(username=username).exists():
            error_message = "이미 사용 중인 아이디입니다."
            return render(request, 'register.html', {'error_message': error_message})

        user = User.objects.create_user(username=username, password=password)
        user = authenticate(request, username=username, password=password)
        login(request, user)
        return redirect('login')  # 회원가입 후 로그인 페이지로 리디렉션

    return render(request, 'register.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  # 로그인 후 홈으로 리디렉션
        else:
            error_message = "아이디 또는 비밀번호가 잘못되었습니다."
            return render(request, 'login.html', {'error_message': error_message})
    return render(request, 'login.html')

def logout_view(request):
    logout(request)
    return redirect('home')

from typing import Optional, Tuple, List
from konlpy.tag import Okt
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from pharmtracker import settings
import xml.etree.ElementTree as ET
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from django.utils.html import escape
from urllib.parse import quote
import asyncio
import aiohttp

# Initialize the Okt tokenizer
okt = Okt()

# 전역 변수로 vectorstore 선언
vectorstore = None

# RAG 설정
llm = ChatOpenAI(temperature=0.1, openai_api_key=openai.api_key, model="gpt-4o-mini")
cache_dir = LocalFileStore("./.cache/practice/")
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

def load_pdf_chunks(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    # PDF 문서를 로드합니다.
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 청크로 나누기
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(pages)
    return chunks

def add_pdf_to_vectorstore(pdf_path: str, vectorstore):
    # PDF 내용을 청크로 로드합니다.
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 텍스트 분할기를 사용하여 문서를 청크로 나눔
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,  # 청크의 크기 설정
        chunk_overlap=100  # 청크 간 중복 설정
    )
    split_docs = text_splitter.split_documents(documents)

    # 문서에서 텍스트를 추출
    texts = [doc.page_content for doc in split_docs]

    # 문서를 벡터 저장소에 추가
    vectorstore.add_texts(texts)
    print(f"벡터 저장소에 {len(texts)}개의 텍스트 청크가 추가되었습니다.")

def add_pdfs_to_vectorstore(pdf_paths: List[str], vectorstore):
    for pdf_path in pdf_paths:
        add_pdf_to_vectorstore(pdf_path, vectorstore)

def initialize_data():
    global vectorstore

    # 벡터 저장소가 초기화되지 않았으면 초기화
    if vectorstore is None:
        vectorstore = FAISS.from_texts(["초기화 문서"], cached_embeddings)

    # 여러 PDF 파일을 벡터 저장소에 추가
    pdf_paths = [
        r"C:\Users\yujinjo\PycharmProjects\graduation-project\pharmtracker\uploads\test_1.pdf",
        r"C:\Users\yujinjo\PycharmProjects\graduation-project\pharmtracker\uploads\twynsta.pdf"
        # 필요에 따라 추가 PDF 파일 경로 추가
    ]
    add_pdfs_to_vectorstore(pdf_paths, vectorstore)

# 앱 시작 시 초기화 호출
initialize_data()

def initialize_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = FAISS.from_texts(["초기화 문서"], cached_embeddings)

# 앱 시작 시 호출
initialize_vectorstore()

# 전역 변수로 의약품 목록 선언
medicine_list = []


def fetch_medicine_list() -> List[str]:
    """
    API를 호출하여 의약품 목록 중 1000개를 가져와 리스트로 반환합니다.
    """
    num_of_rows = 100  # 한 페이지에 가져올 데이터 수
    page_no = 1  # 시작 페이지
    max_pages = 10  # 최대 페이지 수 (1000개 수집 목표)
    medicines = []

    while page_no <= max_pages:
        url = (
            f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/"
            f"getDrugPrdtPrmsnInq05?serviceKey={settings.DUR_API_KEY}&pageNo={page_no}&numOfRows={num_of_rows}&type=xml"
        )
        response = requests.get(url)

        if response.status_code == 200:
            root = ET.fromstring(response.content)
            items = root.findall('.//item')

            for item in items:
                item_name = item.find('ITEM_NAME').text
                if item_name:
                    medicines.append(item_name)

            # print(f"페이지 {page_no}에서 {len(items)}개의 의약품을 가져왔습니다.")

            # 목표 수집량인 1000개에 도달하면 중단
            if len(medicines) >= 1000:
                medicines = medicines[:1000]  # 정확히 1000개로 자르기
                break

            page_no += 1  # 다음 페이지로 이동
        else:
            print(f"API 호출 오류 발생: {response.status_code}")
            break

    return medicines


def initialize_medicine_list():
    """
    의약품 목록을 초기화합니다.
    """
    global medicine_list
    if not medicine_list:  # 초기화되지 않은 경우에만 수행
        medicine_list = fetch_medicine_list()
        print(f"의약품 목록 초기화 완료: {len(medicine_list)}개 항목")

# 앱 시작 시 의약품 목록 초기화
initialize_medicine_list()

def extract_medicine_name_with_llm(question: str) -> Optional[str]:
    """
    Use LLM to extract medicine name from the question.
    """
    # Prompt the LLM to extract potential medicine names
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 의약품 이름을 식별할 수 있는 AI 모델입니다. 
        아래의 질문에서 의약품 이름을 정확히 추출해 주세요. 
        만약 질문에 의약품 이름이 없다면 'None'을 반환하세요.
        """),
        ("human", question)
    ])

    # Run the LLM chain to get the extracted name
    extraction_chain = extraction_prompt | llm
    result = extraction_chain.invoke({"question": question})
    extracted_name = result.content.strip()

    # If the LLM's result is 'None' or empty, we assume no medicine name was found
    if extracted_name.lower() in ['none', '']:
        return None

    return extracted_name

def preprocess_dur_data(xml_data):
    # Parse the XML data
    root = ET.fromstring(xml_data)
    items = root.findall('.//item')
    processed_data = []

    # print(f"API 응답에서 {len(items)}개의 아이템을 찾았습니다.")

    for item in items:
        processed_item = f"품목기준코드: {item.find('ITEM_SEQ').text if item.find('ITEM_SEQ') is not None else '정보 없음'}\n"
        processed_item += f"품목명: {item.find('ITEM_NAME').text if item.find('ITEM_NAME') is not None else '정보 없음'}\n"
        processed_item += f"제조사: {item.find('ENTP_NAME').text if item.find('ENTP_NAME') is not None else '정보 없음'}\n"
        # processed_item += f"허가일자: {item.find('ITEM_PERMIT_DATE').text if item.find('ITEM_PERMIT_DATE') is not None else '정보 없음'}\n"
        processed_item += f"업종: {item.find('INDUTY_TYPE').text if item.find('INDUTY_TYPE') is not None else '정보 없음'}\n"
        # processed_item += f"위탁 제조사: {item.find('CNSGN_MANUF').text if item.find('CNSGN_MANUF') is not None else '정보 없음'}\n"
        # processed_item += f"기타 일반의약품 코드: {item.find('ETC_OTC_CODE').text if item.find('ETC_OTC_CODE') is not None else '정보 없음'}\n"
        processed_item += f"제형: {item.find('CHART').text if item.find('CHART') is not None else '정보 없음'}\n"
        # processed_item += f"바코드: {item.find('BAR_CODE').text if item.find('BAR_CODE') is not None else '정보 없음'}\n"
        # processed_item += f"원료 이름: {item.find('MATERIAL_NAME').text if item.find('MATERIAL_NAME') is not None else '정보 없음'}\n"
        # processed_item += f"효능 문서 ID: {item.find('EE_DOC_ID').text if item.find('EE_DOC_ID') is not None else '정보 없음'}\n"
        # processed_item += f"사용 방법 문서 ID: {item.find('UD_DOC_ID').text if item.find('UD_DOC_ID') is not None else '정보 없음'}\n"
        # processed_item += f"주의 사항 문서 ID: {item.find('NB_DOC_ID').text if item.find('NB_DOC_ID') is not None else '정보 없음'}\n"
        # processed_item += f"첨부 파일: {item.find('INSERT_FILE').text if item.find('INSERT_FILE') is not None else '정보 없음'}\n"
        processed_item += f"보관 방법: {item.find('STORAGE_METHOD').text if item.find('STORAGE_METHOD') is not None else '정보 없음'}\n"
        processed_item += f"유효 기간: {item.find('VALID_TERM').text if item.find('VALID_TERM') is not None else '정보 없음'}\n"
        # processed_item += f"재심사 대상: {item.find('REEXAM_TARGET').text if item.find('REEXAM_TARGET') is not None else '정보 없음'}\n"
        # processed_item += f"재심사 날짜: {item.find('REEXAM_DATE').text if item.find('REEXAM_DATE') is not None else '정보 없음'}\n"
        processed_item += f"포장 단위: {item.find('PACK_UNIT').text if item.find('PACK_UNIT') is not None else '정보 없음'}\n"
        # processed_item += f"EDI 코드: {item.find('EDI_CODE').text if item.find('EDI_CODE') is not None else '정보 없음'}\n"
        # processed_item += f"문서 텍스트: {item.find('DOC_TEXT').text if item.find('DOC_TEXT') is not None else '정보 없음'}\n"
        # processed_item += f"허가 종류 이름: {item.find('PERMIT_KIND_NAME').text if item.find('PERMIT_KIND_NAME') is not None else '정보 없음'}\n"
        # processed_item += f"회사 번호: {item.find('ENTP_NO').text if item.find('ENTP_NO') is not None else '정보 없음'}\n"
        # processed_item += f"제조 원료 플래그: {item.find('MAKE_MATERIAL_FLAG').text if item.find('MAKE_MATERIAL_FLAG') is not None else '정보 없음'}\n"
        # processed_item += f"신약 분류 이름: {item.find('NEWDRUG_CLASS_NAME').text if item.find('NEWDRUG_CLASS_NAME') is not None else '정보 없음'}\n"
        # processed_item += f"취소 날짜: {item.find('CANCEL_DATE').text if item.find('CANCEL_DATE') is not None else '정보 없음'}\n"
        # processed_item += f"취소 이름: {item.find('CANCEL_NAME').text if item.find('CANCEL_NAME') is not None else '정보 없음'}\n"
        # processed_item += f"변경 날짜: {item.find('CHANGE_DATE').text if item.find('CHANGE_DATE') is not None else '정보 없음'}\n"
        # processed_item += f"마약 종류 코드: {item.find('NARCOTIC_KIND_CODE').text if item.find('NARCOTIC_KIND_CODE') is not None else '정보 없음'}\n"
        # processed_item += f"구분 이름: {item.find('GBN_NAME').text if item.find('GBN_NAME') is not None else '정보 없음'}\n"
        processed_item += f"총 내용량: {item.find('TOTAL_CONTENT').text if item.find('TOTAL_CONTENT') is not None else '정보 없음'}\n"
        # 효능 문서 데이터 추출
        ee_doc_data = item.find('EE_DOC_DATA/DOC/SECTION/ARTICLE/PARAGRAPH')
        ee_doc_text = ee_doc_data.text if ee_doc_data is not None else '정보 없음'
        processed_item += f"효능 문서 데이터: {ee_doc_text}\n"
        # 사용 방법 문서 데이터 추출
        ud_doc_data = item.find('UD_DOC_DATA/DOC/SECTION/ARTICLE/PARAGRAPH')
        ud_doc_text = ud_doc_data.text if ud_doc_data is not None else '정보 없음'
        processed_item += f"사용 방법 문서 데이터: {ud_doc_text}\n"
        # 사용상 주의사항 문서 데이터 추출
        nb_doc_data = item.find('NB_DOC_DATA/DOC/SECTION/ARTICLE/PARAGRAPH')
        nb_doc_text = nb_doc_data.text if nb_doc_data is not None else '정보 없음'
        processed_item += f"주의 사항 문서 데이터: {nb_doc_text}\n"
        # processed_item += f"효능 문서 데이터: {item.find('EE_DOC_DATA').text if item.find('EE_DOC_DATA') is not None else '정보 없음'}\n"
        # processed_item += f"사용 방법 문서 데이터: {item.find('UD_DOC_DATA').text if item.find('UD_DOC_DATA') is not None else '정보 없음'}\n"
        # processed_item += f"주의 사항 문서 데이터: {item.find('NB_DOC_DATA').text if item.find('NB_DOC_DATA') is not None else '정보 없음'}\n"
        # processed_item += f"포장 문서 데이터: {item.find('PN_DOC_DATA').text if item.find('PN_DOC_DATA') is not None else '정보 없음'}\n"
        processed_item += f"주요 성분: {item.find('MAIN_ITEM_INGR').text if item.find('MAIN_ITEM_INGR') is not None else '정보 없음'}\n"
        # processed_item += f"성분 이름: {item.find('INGR_NAME').text if item.find('INGR_NAME') is not None else '정보 없음'}\n"
        # processed_item += f"ATC 코드: {item.find('ATC_CODE').text if item.find('ATC_CODE') is not None else '정보 없음'}\n"
        # processed_item += f"영문 항목 이름: {item.find('ITEM_ENG_NAME').text if item.find('ITEM_ENG_NAME') is not None else '정보 없음'}\n"
        # processed_item += f"영문 회사 이름: {item.find('ENTP_ENG_NAME').text if item.find('ENTP_ENG_NAME') is not None else '정보 없음'}\n"
        # processed_item += f"영문 주요 성분: {item.find('MAIN_INGR_ENG').text if item.find('MAIN_INGR_ENG') is not None else '정보 없음'}\n"
        # processed_item += f"사업자 등록 번호: {item.find('BIZRNO').text if item.find('BIZRNO') is not None else '정보 없음'}\n"

        processed_data.append(processed_item)

    return processed_data


def add_documents_to_vectorstore(documents):
    """
    벡터 저장소에 문서를 추가합니다.
    """
    global vectorstore
    vectorstore.add_texts(documents)
    print(f"벡터 저장소에 {len(documents)}개의 문서가 추가되었습니다.")
    print(f"현재 벡터 저장소의 크기: {vectorstore.index.ntotal}")

async def fetch_data(session, url):
    """
    주어진 URL에서 비동기적으로 데이터를 가져옵니다.
    """
    async with session.get(url) as response:
        return await response.text()

async def load_initial_data_async():
    """
    초기 데이터를 비동기적으로 로드하여 벡터 저장소에 추가합니다.
    """
    encoded_drug_name = urllib.parse.quote("")  # 모든 품목 가져오기
    num_of_rows = 100 # 한 페이지에 100개의 결과
    page_no = 1  # 첫 번째 페이지
    max_pages = 10  # 최대 10페이지
    all_processed_data = []

    async with aiohttp.ClientSession() as session:
        tasks = []

        while page_no <= max_pages:
            dur_api_url = (
                f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/getDrugPrdtPrmsnDtlInq04?serviceKey={settings.DUR_API_KEY}&pageNo={page_no}&numOfRows={num_of_rows}&type=xml&item_name={encoded_drug_name}")

            tasks.append(fetch_data(session, dur_api_url))
            page_no += 1

        responses = await asyncio.gather(*tasks)

        for xml_data in responses:
            processed_data = preprocess_dur_data(xml_data)
            if not processed_data:
                break  # 더 이상 데이터가 없으면 반복 종료

            all_processed_data.extend(processed_data)

    print(f"전처리된 전체 데이터 수: {len(all_processed_data)}")
    add_documents_to_vectorstore(all_processed_data)

def load_initial_data():
    """
    비동기 초기 데이터 로드 함수를 실행합니다.
    """
    asyncio.run(load_initial_data_async())

# 앱 시작 시 초기 데이터 로드
load_initial_data()

def retrieve_relevant_context(question: str) -> Tuple[str, Optional[str]]:
    """
    질문에서 약물 이름을 추출하고, 벡터 저장소에서 관련 문맥을 검색합니다.
    """
    medicine_name = extract_medicine_name_with_llm(question)

    if not medicine_name:
        print("의약품 이름을 추출할 수 없습니다.")
        return "의약품 이름을 질문에서 파악할 수 없습니다. 다시 시도해 주세요.", None

    print(f"추출된 의약품 이름: {medicine_name}")

    # docs_medicine = vectorstore.similarity_search(medicine_name, k=3)
    #print(f"'{medicine_name}'에 대해 검색된 문서 수: {len(docs)}")
    docs_medicine = []
    if medicine_name:
        docs_medicine = vectorstore.similarity_search(medicine_name, k=2)
        print(f"'{medicine_name}'에 대해 검색된 문서 수: {len(docs_medicine)}")

    # PDF 데이터와 관련된 문서 검색
    docs_pdf = vectorstore.similarity_search(question, k=2)
    print(f"질문에 대해 검색된 PDF 문서 수: {len(docs_pdf)}")

    # 두 검색 결과를 결합
    docs = docs_medicine + docs_pdf
    if not docs:
        print("관련 문서를 찾을 수 없습니다.")
        return "관련 문서를 찾을 수 없습니다.", None

    top_item_name = None
    for doc in docs_medicine:
        lines = doc.page_content.splitlines()
        for line in lines:
            if line.startswith("품목명:"):
                top_item_name = line.split(":")[1].strip()
                break

    context = "\n".join([doc.page_content[:1000] for doc in docs])
    return context, top_item_name

def add_button_to_response(result_content: str, drug_name: Optional[str]) -> str:
    """
    결과 응답에 클릭 가능한 버튼을 추가하여 상세 약물 정보를 제공하는 링크를 포함합니다.
    """
    if not drug_name:
        return result_content  # 약물 이름이 없으면 버튼을 추가하지 않음

    encoded_drug_name = quote(drug_name)
    detail_link = f"/drug_list/?drug_name={encoded_drug_name}"

    escaped_drug_name = escape(drug_name)

    button_html = (
        f'<a href="{detail_link}" target="_blank" '
        f'style="display: inline-block; padding: 10px 20px; background-color: #4CAF50; '
        f'color: white; text-align: center; text-decoration: none; font-size: 16px; '
        f'border-radius: 5px;">{escaped_drug_name} 상세 정보</a>'
    )

    return result_content + "\n\n" + button_html

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 중장년층과 노인층을 위한 친절하고 이해하기 쉬운 의약품 정보 제공 도우미입니다. 제공된 의약품 정보를 바탕으로 다음 지침을 따라주세요:
    1. 항상 존댓말을 사용하고, 따뜻하고 공감적인 톤으로 대화하세요.
    2. 의학 용어를 사용할 때는 반드시 쉬운 말로 풀어서 설명해 주세요.
    3. 정보를 제공할 때는 천천히, 명확하게, 그리고 단계적으로 설명해 주세요.
    4. 필요한 경우 반복해서 설명하고, 이해했는지 확인하는 질문을 해 주세요.
    5. 약물 복용과 관련된 주의사항을 강조하고, 의사나 약사와 상담할 것을 권장하세요.
    6. 건강에 도움이 되는 일반적인 조언(예: 규칙적인 운동, 균형 잡힌 식단)도 함께 제공하세요.
    7. 질문의 맥락을 고려하여, 관련된 추가 정보나 조언을 제공하세요.
    8. 답변은 간결하게 5문장 이하로 답변해 주세요.
    9. 의약품 성분을 포함해서 답변해 주세요.
    10. 개조식으로 설명해 주세요.
    아래의 정보를 바탕으로 질문에 정확하고 이해하기 쉽게 답변해 주세요:

    {context}
    """),
    ("human", "{question}")
])

rag_chain = (
    {"context": lambda x: retrieve_relevant_context(x["question"])[0], "question": RunnablePassthrough()}
    | prompt
    | llm
)

@csrf_exempt
@require_http_methods(["GET", "POST"])
def chatbot_view(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8"))
            question = body.get("question")

            if not question:
                return JsonResponse({"error": "질문이 필요합니다."}, status=400)

            context, extracted_drug_name = retrieve_relevant_context(question)
            print(f"검색된 컨텍스트: {context[:1000]}...")  # 디버깅을 위한 컨텍스트 출력

            result = rag_chain.invoke({"question": question, "context": context})
            print(f"생성된 답변: {result.content}")
            print(f"현재 벡터 저장소의 크기: {vectorstore.index.ntotal}")

            # 링크 버튼 추가
            if extracted_drug_name:
                result.content = add_button_to_response(result.content, extracted_drug_name)
                print(f"Added button for drug name: {extracted_drug_name}")

            return JsonResponse({"answer": result.content})

        except json.JSONDecodeError:
            return JsonResponse({"error": "잘못된 JSON 형식입니다."}, status=400)

    elif request.method == "GET":
        return render(request, "chatbot.html")


def drug_list_view(request):
    drug_info = None
    error_message = None

    if request.method == 'POST':
        # 약품 정보를 세션에 저장
        drug = {
            'item_name': request.POST.get('item_name'),
            'entp_name': request.POST.get('entp_name'),
            'spclty_pblc': request.POST.get('spclty_pblc'),
            'prduct_type': request.POST.get('prduct_type'),
            'item_ingr_name': request.POST.get('item_ingr_name'),
            'image_url': request.POST.get('image_url'),
        }

        # 세션에 약품 리스트가 없으면 초기화
        if 'medications' not in request.session:
            request.session['medications'] = []

        # 약품을 세션에 추가
        request.session['medications'].append(drug)
        request.session.modified = True

        messages.success(request, f"{drug['item_name']}이(가) 내 약품 목록에 추가되었습니다.")

        return redirect('drug_list')  # 약품 목록 페이지로 리디렉션

    elif request.method == 'GET':
        # 기존 GET 요청 처리 코드
        api_key = settings.DUR_API_KEY  # 공공데이터포털에서 발급받은 인코딩된 인증 키
        drug_name = request.GET.get('drug_name')  # 사용자 입력을 받음

        if drug_name:
            encoded_drug_name = urllib.parse.quote(drug_name)
            url = f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/getDrugPrdtPrmsnInq05?serviceKey={api_key}&pageNo=1&numOfRows=10&type=xml&item_name={encoded_drug_name}"
            response = requests.get(url, verify=False)  # SSL 검증 비활성화

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "lxml-xml")

                drug_list = []

                for item in soup.find_all("item"):
                    item_name = item.find("ITEM_NAME").text if item.find("ITEM_NAME") else ""
                    if drug_name.lower() in item_name.lower():
                        entp_name = item.find("ENTP_NAME").text if item.find("ENTP_NAME") else None
                        spclty_pblc = item.find("SPCLTY_PBLC").text if item.find("SPCLTY_PBLC") else None
                        prduct_type = item.find("PRDUCT_TYPE").text if item.find("PRDUCT_TYPE") else None
                        item_ingr_name = item.find("ITEM_INGR_NAME").text if item.find("ITEM_INGR_NAME") else None
                        item_seq = item.find("ITEM_SEQ").text if item.find("ITEM_SEQ") else None

                        detail_link = f"/drug_detail/{item_seq}/"  # 새로운 상세 정보 페이지로의 링크
                        image_url = get_drug_image(item_name)  # 약품 이미지 URL 가져오기

                        drug_list.append({
                            "item_name": item_name,
                            "entp_name": entp_name,
                            "spclty_pblc": spclty_pblc,
                            "prduct_type": prduct_type,
                            "item_ingr_name": item_ingr_name,
                            "detail_link": detail_link,
                            "image_url": image_url  # 약품 이미지 URL 추가
                        })

                if not drug_list:
                    error_message = f"'{drug_name}'에 대한 의약품 정보를 찾을 수 없습니다."
                else:
                    drug_info = drug_list
            else:
                error_message = f"API 요청 실패: {response.status_code} - {response.text}"

        # 세션에서 저장된 약품 목록 가져오기
        medications = request.session.get('medications', [])
        # 저장된 약품의 이름 목록을 집합으로 생성
        saved_item_names = set(med['item_name'] for med in medications)

        return render(request, 'drug_list.html', {
            'drug_info': drug_info,
            'error_message': error_message,
            'saved_item_names': saved_item_names,  # 추가된 부분
        })

    return render(request, 'drug_list.html', {'drug_info': drug_info, 'error_message': error_message})
def get_drug_image(drug_name):
    api_key = settings.DUR_API_KEY  # 공공데이터포털에서 발급받은 인코딩된 인증 키
    encoded_drug_name = urllib.parse.quote(drug_name)
    url = f"http://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService01/getMdcinGrnIdntfcInfoList01?serviceKey={api_key}&item_name={encoded_drug_name}&pageNo=1&numOfRows=3&type=xml"
    response = requests.get(url, verify=False)  # SSL 검증 비활성화

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "lxml-xml")
        item = soup.find("item")
        if item:
            image_url = item.find("ITEM_IMAGE").text if item.find("ITEM_IMAGE") else None
            print(f"Extracted Image URL: {image_url}")  # 디버깅용 출력
            return image_url
    return None

def drug_detail_view(request, item_seq):
    api_key = settings.DUR_API_KEY  # 공공데이터포털에서 발급받은 인코딩된 인증 키
    encoded_item_seq = urllib.parse.quote(item_seq)
    url = f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/getDrugPrdtPrmsnDtlInq04?serviceKey={api_key}&item_seq={encoded_item_seq}&type=xml"
    response = requests.get(url, verify=False)  # SSL 검증 비활성화

    drug_detail = {}
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "lxml-xml")
        item = soup.find("item")
        if item:
            def extract_text(element):
                return ''.join(element.findAll(text=True)) if element else None

            drug_detail = {
                "item_name": extract_text(item.find("ITEM_NAME")),
                "entp_name": extract_text(item.find("ENTP_NAME")),
                "efficacy": extract_text(item.find("EE_DOC_DATA")),
                "usage_dosage": extract_text(item.find("UD_DOC_DATA")),
                "precautions": extract_text(item.find("NB_DOC_DATA")),
                "image_url": get_drug_image(extract_text(item.find("ITEM_NAME")))
            }
    else:
        drug_detail = None

    return render(request, 'drug_detail.html', {'drug_detail': drug_detail})

def get_drug_info(drug_name, api_key):
    print(f"Fetching drug info for: {drug_name}")
    encoded_drug_name = urllib.parse.quote(drug_name)
    url = f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/getDrugPrdtPrmsnInq05?serviceKey={api_key}&pageNo=1&numOfRows=10&type=xml&item_name={encoded_drug_name}"
    response = requests.get(url, verify=False)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "lxml-xml")
        items = soup.find_all("item")
        drugs = []
        for item in items:
            drug = {
                'ITEM_NAME': item.find("ITEM_NAME").text if item.find("ITEM_NAME") else "",
                'ITEM_SEQ': item.find("ITEM_SEQ").text if item.find("ITEM_SEQ") else "",
                'ENTP_NAME': item.find("ENTP_NAME").text if item.find("ENTP_NAME") else "",
                'SPCLTY_PBLC': item.find("SPCLTY_PBLC").text if item.find("SPCLTY_PBLC") else "",
                'PRDUCT_TYPE': item.find("PRDUCT_TYPE").text if item.find("PRDUCT_TYPE") else "",
                'ITEM_INGR_NAME': item.find("ITEM_INGR_NAME").text if item.find("ITEM_INGR_NAME") else ""
            }
            print(f"Drug found: {drug}")
            drugs.append(drug)
        return drugs
    else:
        print("Failed to fetch drug info.")
        return None


def ocr_view(request):
    return render(request, 'ocr.html')

def put_text(img, text, x, y, font_size):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x, y), font, font_size, (0, 255, 0), 2)
    return img

irrelevant_words = ["밀리그램", "mg", "정", "캡슐", "tablet", "softgel", "연질캡슐"]
irrelevant_pattern = re.compile(r'\d+밀리그램|\d+mg')
brackets_pattern = re.compile(r'\((.*?)\)')  # 괄호 안의 텍스트를 추출하는 정규식

def match_single_words(ocr_text, drug_name):
    if ocr_text not in irrelevant_words and not irrelevant_pattern.match(ocr_text):
        if ocr_text in drug_name:
            return True  # 매칭되는 단어가 하나라도 있으면 True 반환
    return False

@csrf_exempt
def ocr_process(request):
    if request.method == 'POST':
        try:
            print("Processing started...")

            path = None

            # FormData 방식 (앨범 업로드)
            if 'image' in request.FILES:
                image_file = request.FILES['image']
                path = os.path.join('uploads', image_file.name)
                with open(path, 'wb') as f:
                    for chunk in image_file.chunks():
                        f.write(chunk)
                print(f"Image uploaded and saved at {path}")

            # JSON 방식 (카메라 캡처)
            elif request.body:
                try:
                    body_data = json.loads(request.body.decode('utf-8'))
                    image_data = body_data.get('image_data')
                    if image_data:
                        image_data = base64.b64decode(image_data)
                        img = Image.open(BytesIO(image_data))
                        path = os.path.join('uploads', f'{uuid.uuid4()}.jpg')
                        img.save(path, 'JPEG')
                        print(f"Captured image saved at {path}")
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Failed to decode JSON or process image data: {str(e)}")
                    return JsonResponse({'error': 'Invalid image data.'}, status=400)
            else:
                print("No image provided.")
                return JsonResponse({'error': 'No image provided.'}, status=400)

            # 파일이 제대로 업로드되었는지 확인
            if not path or not os.path.exists(path):
                print("File upload failed.")
                return JsonResponse({'error': 'File upload failed.'}, status=500)

            print(f"File exists at {path}. Proceeding with OCR...")

            files = [('file', open(path, 'rb'))]

            request_json = {
                'images': [{'format': 'jpg', 'name': 'demo'}],
                'requestId': str(uuid.uuid4()),
                'version': 'V2',
                'timestamp': int(round(time.time() * 1000))
            }

            payload = {'message': json.dumps(request_json).encode('UTF-8')}

            headers = {
                'X-OCR-SECRET': secret_key,
            }

            response = requests.post(api_url, headers=headers, data=payload, files=files)
            result = response.json()

            print("OCR API response received.")

            # 이미지 파일 읽기
            img = cv2.imread(path)

            # 이미지 로드 확인
            if img is None:
                print("Failed to read image file.")
                return JsonResponse({'error': 'Failed to read image file.'}, status=500)

            roi_img = img.copy()

            # 인식된 모든 텍스트 추출
            recognized_texts = []

            for field in result['images'][0]['fields']:
                text = field['inferText']
                vertices_list = field['boundingPoly']['vertices']
                pts = [tuple(vertice.values()) for vertice in vertices_list]
                topLeft = [int(_) for _ in pts[0]]
                topRight = [int(_) for _ in pts[1]]
                bottomRight = [int(_) for _ in pts[2]]
                bottomLeft = [int(_) for _ in pts[3]]

                cv2.line(roi_img, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(roi_img, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(roi_img, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(roi_img, bottomLeft, topLeft, (0, 255, 0), 2)
                roi_img = put_text(roi_img, text, topLeft[0], topLeft[1] - 10, font_size=30)

                recognized_texts.append(text)  # 모든 인식된 텍스트 추가

            print(f"Recognized Texts: {recognized_texts}")

            # 처리된 이미지를 저장하여 프론트엔드에 제공
            processed_image_path = os.path.join('uploads', 'processed_' + os.path.basename(path))
            cv2.imwrite(processed_image_path, roi_img)
            print(f"Processed image saved at {processed_image_path}")

            # 검색 결과가 있는 텍스트 필터링
            valid_texts = []
            korean_pattern = re.compile(r'^[가-힣0-9]+$')

            for text in recognized_texts:
                # irrelevant_words에 있는 단어가 포함된 텍스트는 제외
                if any(word in text for word in irrelevant_words):
                    print(f"Skipping '{text}' because it contains one of the irrelevant words")
                    continue  # irrelevant_words에 포함된 단어가 있는 텍스트는 제외

                # 한글 또는 숫자로 이루어진 텍스트인지 확인 (두 글자 이상)
                if korean_pattern.match(text) and len(text) >= 2:
                    print(f"Possible drug name from OCR: {text}")

                    # 의약품 정보를 검색하기 위한 API URL 구성
                    encoded_text = urllib.parse.quote(text)
                    url = f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/getDrugPrdtPrmsnInq05?serviceKey={settings.DUR_API_KEY}&type=xml&item_name={encoded_text}"
                    response = requests.get(url)

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, "html.parser")
                        # API 응답에서 item_name 태그를 찾아 의약품명 확인
                        item_name_tags = soup.find_all("item_name")
                        if item_name_tags:
                            for item_name in item_name_tags:
                                drug_name = item_name.get_text().strip()

                                # 괄호 안의 텍스트를 추출하여 확인
                                bracketed_texts = brackets_pattern.findall(drug_name)
                                skip = False
                                for bracketed_text in bracketed_texts:
                                    if text in bracketed_text:
                                        print(f"Skipping '{drug_name}' because '{text}' is inside the bracket")
                                        skip = True
                                        break

                                if skip:
                                    continue

                                print(f"API returned drug name: {drug_name}")

                                if match_single_words(text, drug_name):
                                    print(f"Single word match found with '{drug_name}'")
                                    valid_texts.append(text)
                    else:
                        print(f"API request failed with status {response.status_code} for {text}")

            # 최종 valid_texts 반환
            print(f"Valid Texts: {valid_texts}")

            # 검색 결과가 있는 첫 번째 유효 텍스트로 리다이렉트 URL 생성
            if valid_texts:
                redirect_url = request.build_absolute_uri(f'/drug_list/?drug_name={urllib.parse.quote(valid_texts[0])}')
                return JsonResponse({'redirect': redirect_url})

            print("No valid drug information found.")
            return JsonResponse({'error': 'No valid drug information found in OCR result.'}, status=404)

        except Exception as e:
            print(f"Error occurred during OCR processing: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

    print("Invalid request method.")
    return JsonResponse({'error': 'Invalid request'}, status=400)
def put_text(img, text, x, y, font_size):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x, y), font, font_size, (0, 255, 0), 2)
    return img

def get_articles(query='', page=1):
    if query:
        url = f"https://www.kpanews.co.kr/article/search4.asp?top_keyword={query}&page={page}"
    else:
        url = f"https://www.kpanews.co.kr/article/list.asp?page={page}"

    response = requests.get(url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    articles = []

    for item in soup.select('.lst_article1 ul li'):
        try:
            title = item.select_one('.subj').text.strip()
            summary = item.select_one('.t1').text.strip() if item.select_one('.t1') else ''
            date = item.select_one('.botm span').text.strip()
            author = item.select('.botm span')[1].text.strip() if len(item.select('.botm span')) > 1 else ''
            link = "https://www.kpanews.co.kr/article/" + item.a['href']

            articles.append({
                'title': title,
                'summary': summary,
                'date': date,
                'author': author,
                'link': link
            })
        except Exception as e:
            print(f"Error while processing article: {str(e)}")

    return articles


def news_view(request):
    query = request.GET.get('query', '')
    page = int(request.GET.get('page', 1))
    articles = get_articles(query, page)
    return render(request, 'news.html', {'articles': articles, 'query': query, 'page': page})


def news_summary_view(request, article_id):
    page = int(request.GET.get('page', 1))
    articles = get_articles(page=page)

    # Ensure article_id is within the range of articles list
    if 0 <= article_id < len(articles):
        article = articles[article_id]
    else:
        return render(request, 'error.html', {'message': 'Article not found'})

    article_response = requests.get(article['link'])
    article_response.encoding = 'utf-8'
    article_soup = BeautifulSoup(article_response.text, 'html.parser')

    article_content = article_soup.select_one('.view_con_t')
    if article_content:
        article_text = article_content.get_text(strip=True)
    else:
        article_text = article_soup.get_text(strip=True)

    max_length = 10000  # 길이 증가
    article_text = article_text[:max_length]

    try:
        # LangChain을 사용하여 요약 생성
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes Korean news articles."},
            {"role": "user", "content": f"다음 기사를 자세히 요약해주세요 (약 300-400자):\n\n{article_text}"}
        ]

        response = llm(messages=messages)
        full_summary = response.content.strip()
    except Exception as e:
        full_summary = f"요약 생성 중 오류 발생: {str(e)}"

    article['full_summary'] = full_summary
    return render(request, 'news_summary.html', {'article': article})


def get_pharmacies(search_query='', page=1, num_of_rows=10):
    base_url = "http://apis.data.go.kr/B552657/ErmctInsttInfoInqireService/getParmacyListInfoInqire"
    params = {
        'serviceKey': settings.DUR_API_KEY,
        'QT': '1',  # Query type parameter
        'ORD': 'NAME',
        'pageNo': page,
        'numOfRows': num_of_rows,
    }

    # 검색 쿼리를 URL 디코딩
    search_query = unquote(search_query).strip()

    # 지역과 약국명을 구분
    name_part = ''
    region_part = ''

    # 검색 쿼리가 "약국명 (주소)" 형식일 경우
    if '(' in search_query and search_query.endswith(')'):
        name_part, full_address = search_query.split('(', 1)
        name_part = name_part.strip()
        full_address = full_address.rsplit(')', 1)[0].strip()

        # 주소에서 지역만 추출 (예: 서울특별시)
        if ' ' in full_address:
            region_part = full_address.split(' ')[0]

    # 사용자가 입력한 쿼리가 지역명으로 시작하면 지역으로 설정
    elif ' ' in search_query:
        potential_region = search_query.split(' ')[0]
        # 한국의 일반적인 시/도 이름에 해당하는 경우
        if potential_region.endswith('시') or potential_region.endswith('도'):
            region_part = potential_region
            name_part = search_query.replace(region_part, '').strip()
        else:
            name_part = search_query

    else:
        name_part = search_query

    # 파라미터 설정
    if name_part:
        params['QN'] = name_part  # 약국명으로 검색
    if region_part:
        params['Q0'] = region_part  # 지역으로 검색

    # URL 생성
    url = (f"{base_url}?serviceKey={params['serviceKey']}"
           f"&QT={params['QT']}&ORD={params['ORD']}&pageNo={params['pageNo']}"
           f"&numOfRows={params['numOfRows']}")

    if 'Q0' in params:
        url += f"&Q0={params['Q0']}"
    if 'QN' in params:
        url += f"&QN={params['QN']}"

    response = requests.get(url, verify=False)
    response.encoding = 'utf-8'

    if response.status_code != 200:
        return []

    try:
        tree = ElementTree.fromstring(response.content)
        body = tree.find('body')
        if body is None:
            return []

        items = body.find('items')
        if items is None:
            return []

        pharmacies = []
        for item in items.findall('item'):
            pharmacy = {
                'name': item.findtext('dutyName', default='N/A'),
                'address': item.findtext('dutyAddr', default='N/A'),
                'building': item.findtext('dutyMapimg', default='N/A'),
                'telephone': item.findtext('dutyTel1', default='N/A'),
                'hours': {
                    'mon': (item.findtext('dutyTime1s', default='N/A'), item.findtext('dutyTime1c', default='N/A')),
                    'tue': (item.findtext('dutyTime2s', default='N/A'), item.findtext('dutyTime2c', default='N/A')),
                    'wed': (item.findtext('dutyTime3s', default='N/A'), item.findtext('dutyTime3c', default='N/A')),
                    'thu': (item.findtext('dutyTime4s', default='N/A'), item.findtext('dutyTime4c', default='N/A')),
                    'fri': (item.findtext('dutyTime5s', default='N/A'), item.findtext('dutyTime5c', default='N/A')),
                    'sat': (item.findtext('dutyTime6s', default='N/A'), item.findtext('dutyTime6c', default='N/A')),
                },
                'lat': item.findtext('wgs84Lat', default='N/A'),
                'lon': item.findtext('wgs84Lon', default='N/A'),
            }
            pharmacies.append(pharmacy)

        return pharmacies

    except ElementTree.ParseError as e:
        print(f"XML 파싱 오류: {str(e)}")
        return []


def get_all_pharmacies_for_autocomplete():
    # Try to get the data from cache first
    cache_key = 'all_pharmacies_autocomplete'
    all_pharmacies = cache.get(cache_key)

    if all_pharmacies is None:
        # If not in cache, fetch from API
        all_pharmacies = []
        page = 1
        while True:
            pharmacies = get_pharmacies(page=page, num_of_rows=100)
            if not pharmacies:
                break
            all_pharmacies.extend([(p['name'], p['address']) for p in pharmacies])
            page += 1

        # Store in cache for 1 day
        cache.set(cache_key, all_pharmacies, 60 * 60 * 24)

    return all_pharmacies


def pharmacy_list_view(request):
    search_query = request.GET.get('search_query', '')
    search_query = unquote(search_query)  # URL 디코딩

    page = int(request.GET.get('page', 1))
    pharmacies = get_pharmacies(search_query, page)
    all_pharmacies = get_all_pharmacies_for_autocomplete()

    # 검색 결과의 첫 번째 약국 좌표 설정
    if pharmacies:
        first_pharmacy = pharmacies[0]
        search_center_lat = float(first_pharmacy['lat']) if first_pharmacy['lat'] != 'N/A' else 37.5665
        search_center_lon = float(first_pharmacy['lon']) if first_pharmacy['lon'] != 'N/A' else 126.9780
    else:
        # 검색 결과가 없는 경우 기본값 설정 (예: 서울시청)
        search_center_lat = 37.5665
        search_center_lon = 126.9780

    context = {
        'pharmacies': pharmacies,
        'search_query': search_query,
        'all_pharmacies': all_pharmacies,
        'search_center_lat': search_center_lat,
        'search_center_lon': search_center_lon
    }
    return render(request, 'pharmacy_list.html', context)
def get_all_pharmacies_for_autocomplete():
    # Try to get the data from cache first
    cache_key = 'all_pharmacies_autocomplete'
    all_pharmacies = cache.get(cache_key)

    if all_pharmacies is None:
        # If not in cache, fetch from API
        all_pharmacies = []
        page = 1
        while True:
            pharmacies = get_pharmacies(page=page, num_of_rows=100)
            if not pharmacies:
                break
            all_pharmacies.extend([(p['name'], p['address']) for p in pharmacies])
            page += 1

        # Store in cache for 1 day
        cache.set(cache_key, all_pharmacies, 60 * 60 * 24)

    return all_pharmacies


def pharmacy_list_view(request):
    search_query = request.GET.get('search_query', '')
    search_query = unquote(search_query)  # URL 디코딩

    page = int(request.GET.get('page', 1))
    pharmacies = get_pharmacies(search_query, page)
    all_pharmacies = get_all_pharmacies_for_autocomplete()

    # 검색 결과의 첫 번째 약국 좌표 설정
    if pharmacies:
        first_pharmacy = pharmacies[0]
        search_center_lat = float(first_pharmacy['lat']) if first_pharmacy['lat'] != 'N/A' else 37.5665
        search_center_lon = float(first_pharmacy['lon']) if first_pharmacy['lon'] != 'N/A' else 126.9780
    else:
        # 검색 결과가 없는 경우 기본값 설정 (예: 서울시청)
        search_center_lat = 37.5665
        search_center_lon = 126.9780

    context = {
        'pharmacies': pharmacies,
        'search_query': search_query,
        'all_pharmacies': all_pharmacies,
        'search_center_lat': search_center_lat,
        'search_center_lon': search_center_lon
    }
    return render(request, 'pharmacy_list.html', context)


def drug_interaction_view(request):
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        # AJAX 요청 처리 (자동완성)
        drug_name = request.GET.get('term', '')
        drugs = get_drug_info(drug_name, settings.DUR_API_KEY)
        suggestions = [{'label': drug['ITEM_NAME'], 'value': drug['ITEM_SEQ']} for drug in drugs] if drugs else []
        return JsonResponse(suggestions, safe=False)

    registered_drugs = request.session.get('registered_drugs', [])
    comparison_result = None
    error_message = None

    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'register':
            item_seq = request.POST.get('item_seq')
            item_name = request.POST.get('item_name')
            if item_seq and item_name:
                drug_info = get_drug_detail(item_seq, settings.DUR_API_KEY)
                if drug_info:
                    registered_drugs.append(drug_info)
                    request.session['registered_drugs'] = registered_drugs
        elif action == 'remove':
            item_seq = request.POST.get('item_seq')
            registered_drugs = [drug for drug in registered_drugs if drug['item_seq'] != item_seq]
            request.session['registered_drugs'] = registered_drugs
        elif action == 'compare':
            comparison_result = compare_drugs(registered_drugs, settings.DUR_API_KEY)

    if request.method == 'GET':
        drug_name = request.GET.get('drug_name')
        if drug_name:
            drugs = get_drug_info(drug_name, settings.DUR_API_KEY)
            if not drugs:
                error_message = f"'{drug_name}'에 대한 의약품 정보를 찾을 수 없습니다."

    context = {
        'registered_drugs': registered_drugs,
        'comparison_result': comparison_result,
        'error_message': error_message
    }
    return render(request, 'drug_interaction.html', context)

def get_drug_detail(item_seq, api_key):
    encoded_item_seq = urllib.parse.quote(item_seq)
    url = f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/getDrugPrdtPrmsnDtlInq04?serviceKey={api_key}&item_seq={encoded_item_seq}&type=xml"
    response = requests.get(url, verify=False)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "lxml-xml")
        item = soup.find("item")
        if item:
            drug_detail = {
                "item_seq": item_seq,
                "item_name": item.find("ITEM_NAME").text if item.find("ITEM_NAME") else "",
                "entp_name": item.find("ENTP_NAME").text if item.find("ENTP_NAME") else "",
                "material_name": item.find("MATERIAL_NAME").text if item.find("MATERIAL_NAME") else "",
                "image_url": get_drug_image(item.find("ITEM_NAME").text) if item.find("ITEM_NAME") else None
            }
            print(f"Drug detail found: {drug_detail}")
            return drug_detail
    return None


def compare_drugs(registered_drugs, api_key):
    comparison_result = []
    for i, drug1 in enumerate(registered_drugs):
        for drug2 in registered_drugs[i + 1:]:
            interaction_results = {
                'contraindication': check_drug_interaction(drug1['item_seq'], drug2['item_seq'], api_key),
                'age_restriction': check_age_restriction(drug1['item_seq'], drug2['item_seq'], api_key),
                'pregnancy_restriction': check_pregnancy_restriction(drug1['item_seq'], drug2['item_seq'], api_key),
                'elderly_caution': check_elderly_caution(drug1['item_seq'], drug2['item_seq'], api_key),
                'dosage_caution': check_dosage_caution(drug1['item_seq'], drug2['item_seq'], api_key),
                'duration_caution': check_duration_caution(drug1['item_seq'], drug2['item_seq'], api_key)
            }

            comparison_result.append({
                'drug1': drug1['item_name'],
                'drug2': drug2['item_name'],
                'interactions': interaction_results
            })

    return comparison_result


def fetch_interaction_data(url1, url2):
    interactions = set()

    for url in [url1, url2]:
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "lxml-xml")
            items = soup.find_all("item")
            for item in items:
                type_name = item.find("TYPE_NAME").text if item.find("TYPE_NAME") else "알 수 없는 유형"
                mixture_ingr_eng = item.find("MIXTURE_INGR_ENG_NAME").text if item.find("MIXTURE_INGR_ENG_NAME") else "알 수 없는 성분"
                content = item.find("PROHBT_CONTENT").text if item.find("PROHBT_CONTENT") else "상세 내용 없음"
                interaction = f"{type_name} ({mixture_ingr_eng}): {content}"
                interactions.add(interaction)
        else:
            interactions.add("정보를 가져오는 데 실패했습니다.")

    return " | ".join(interactions) if interactions else "해당 정보 없음"

def check_drug_interaction(item_seq1, item_seq2, api_key):
    url1 = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getUsjntTabooInfoList03?serviceKey={api_key}&itemSeq={item_seq1}&itemSeq2={item_seq2}&type=xml"
    url2 = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getUsjntTabooInfoList03?serviceKey={api_key}&itemSeq={item_seq2}&itemSeq2={item_seq1}&type=xml"
    return fetch_interaction_data(url1, url2)

def check_age_restriction(item_seq1, item_seq2, api_key):
    url1 = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getSpcifyAgrdeTabooInfoList03?serviceKey={api_key}&itemSeq={item_seq1}&itemSeq2={item_seq2}&type=xml"
    url2 = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getSpcifyAgrdeTabooInfoList03?serviceKey={api_key}&itemSeq={item_seq2}&itemSeq2={item_seq1}&type=xml"
    return fetch_interaction_data(url1, url2)

def check_pregnancy_restriction(item_seq1, item_seq2, api_key):
    url1 = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getPwnmTabooInfoList03?serviceKey={api_key}&itemSeq={item_seq1}&itemSeq2={item_seq2}&type=xml"
    url2 = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getPwnmTabooInfoList03?serviceKey={api_key}&itemSeq={item_seq2}&itemSeq2={item_seq1}&type=xml"
    return fetch_interaction_data(url1, url2)

def check_elderly_caution(item_seq1, item_seq2, api_key):
    url1 = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getOdsnAtentInfoList03?serviceKey={api_key}&itemSeq={item_seq1}&itemSeq2={item_seq2}&type=xml"
    url2 = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getOdsnAtentInfoList03?serviceKey={api_key}&itemSeq={item_seq2}&itemSeq2={item_seq1}&type=xml"
    return fetch_interaction_data(url1, url2)

def check_dosage_caution(item_seq1, item_seq2, api_key):
    url1 = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getCpctyAtentInfoList03?serviceKey={api_key}&itemSeq={item_seq1}&itemSeq2={item_seq2}&type=xml"
    url2 = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getCpctyAtentInfoList03?serviceKey={api_key}&itemSeq={item_seq2}&itemSeq2={item_seq1}&type=xml"
    return fetch_interaction_data(url1, url2)

def check_duration_caution(item_seq1, item_seq2, api_key):
    url1 = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getMdctnPdAtentInfoList03?serviceKey={api_key}&itemSeq={item_seq1}&itemSeq2={item_seq2}&type=xml"
    url2 = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getMdctnPdAtentInfoList03?serviceKey={api_key}&itemSeq={item_seq2}&itemSeq2={item_seq1}&type=xml"
    return fetch_interaction_data(url1, url2)


def extract_medicine_from_pdf(pdf_path: str) -> List[str]:
    # PDF 문서를 청크로 나누기
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 청크로 나누기
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)

    medicine_list = []
    # 의약품 패턴 예시 - 의약품 이름은 영어 단어로 가정
    medicine_pattern = re.compile(r'\b[A-Za-z]+\b')

    for chunk in chunks:
        # 청크에서 의약품 이름을 찾기
        matches = medicine_pattern.findall(chunk.page_content)
        medicine_list.extend(matches)

    return medicine_list


def add_medicine_buttons_to_response(result_content: str, medicines: List[str]) -> str:
    # 의약품 이름마다 버튼을 생성
    button_html = ""
    for medicine in medicines:
        encoded_medicine_name = quote(medicine)
        detail_link = f"/drug_list/?drug_name={encoded_medicine_name}"
        button_html += (
            f'<a href="{detail_link}" target="_blank" '
            f'style="display: inline-block; padding: 10px 20px; background-color: #4CAF50; '
            f'color: white; text-align: center; text-decoration: none; font-size: 16px; '
            f'border-radius: 5px; margin-right: 5px;">{escape(medicine)} 상세 정보</a>'
        )

    return result_content + "\n\n" + button_html


def chatbot_view(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8"))
            question = body.get("question")

            if not question:
                return JsonResponse({"error": "질문이 필요합니다."}, status=400)

            context, extracted_drug_name = retrieve_relevant_context(question)

            # PDF 파일에서 의약품 정보를 추출
            pdf_medicines = extract_medicine_from_pdf("/path/to/pdf")

            # 챗봇 답변 생성
            result = rag_chain.invoke({"question": question, "context": context})

            # 추출된 의약품 이름으로 버튼을 생성하여 추가
            result.content = add_medicine_buttons_to_response(result.content, pdf_medicines)

            return JsonResponse({"answer": result.content})

        except json.JSONDecodeError:
            return JsonResponse({"error": "잘못된 JSON 형식입니다."}, status=400)

        
def news_view(request):
    query = request.GET.get('query','')
    page = request.GET.get('page', 1)
    articles = get_articles(page)
    if query:  # 검색어가 있을 경우
        # 검색어가 기사 제목이나 요약에 포함된 기사만 필터링
        articles = [article for article in articles if query.lower() in article['title'].lower() or query.lower() in article['summary'].lower()]
    return render(request, 'news.html', {'articles': articles, 'query': query, 'page': page})