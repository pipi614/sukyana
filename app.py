import streamlit as st
import cv2
import numpy as np
import easyocr
import json
import pandas as pd

# ==========================================
# 0. ページ設定 & タッチ操作向けCSS（iPad最適化）
# ==========================================
st.set_page_config(
    page_title="iPad対応 AI単語帳スキャナー", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ボタンの高さや文字サイズを大きくしてタッチしやすくするカスタムスタイル
st.markdown("""
<style>
    .stButton>button {
        height: 3em;
        font-size: 1.2rem !important;
        border-radius: 12px;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# EasyOCRリーダーのキャッシュ化（アプリ起動高速化・無料リソース節約）
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['ja', 'en'])

reader = load_ocr_reader()

# ==========================================
# 1. セッション状態の初期化
# ==========================================
if "card_data" not in st.session_state:
    st.session_state.card_data = [
        {"id": 1, "front": "photosynthesis", "back": "光合成", "correct": 0, "total": 0},
        {"id": 2, "front": "mitochondria", "back": "ミトコンドリア", "correct": 0, "total": 0},
    ]

if "notes" not in st.session_state:
    st.session_state.notes = ""

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "show_answer" not in st.session_state:
    st.session_state.show_answer = False

# ==========================================
# 2. 画像処理 & OCR関数（OpenCV + EasyOCR）
# ==========================================
def process_image_and_extract_text(image_bytes):
    """画像から緑色マーカー領域を抜き出してOCR実行"""
    # バイトデータからOpenCV画像オブジェクトへ変換
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        return []

    # HSV変換して緑色マーカー部分を抽出
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cropped_regions = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # 小さなノイズを除外
            x, y, w, h = cv2.boundingRect(cnt)
            cropped = image[y:y+h, x:x+w]
            cropped_regions.append((y, cropped))

    # 上から順にソート
    cropped_regions.sort(key=lambda item: item[0])

    extracted_cards = []
    for idx, (_, crop) in enumerate(cropped_regions):
        results = reader.readtext(crop)
        text = " ".join([res[1] for res in results]).strip()
        if text:
            extracted_cards.append({
                "id": len(st.session_state.card_data) + idx + 1,
                "front": text,
                "back": "",  # 意味は手書きまたは手入力
                "correct": 0,
                "total": 0
            })
    return extracted_cards

# ==========================================
# 3. サイドバー・画面切替
# ==========================================
st.sidebar.title("メニュー")
app_mode = st.sidebar.radio("画面モードを選択", ["画面①: スキャン・テキスト編集", "画面②: 単語カード学習"])

# ==========================================
# 画面①：スキャン・テキスト編集画面
# ==========================================
if app_mode == "画面①: スキャン・テキスト編集":
    st.header("📱 画面①: スキャン＆テキスト編集")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("スキャン画像入力")
        input_type = st.radio("入力方法を選択", ["ファイル選択", "iPadカメラで撮影"], horizontal=True)
        
        uploaded_image = None
        if input_type == "ファイル選択":
            uploaded_image = st.file_uploader("画像をアップロード", type=["jpg", "png", "jpeg"])
        else:
            uploaded_image = st.camera_input("iPadのカメラでパシャリ")

        if uploaded_image is not None:
            st.image(uploaded_image, caption="取り込み画像", use_container_width=True)
            
            if st.button("🔍 画像からマーカー文字を自動抽出", use_container_width=True, type="primary"):
                with st.spinner("マーカー部分を解析中..."):
                    new_cards = process_image_and_extract_text(uploaded_image)
                    if new_cards:
                        st.session_state.card_data.extend(new_cards)
                        st.success(f"{len(new_cards)} 件の単語を抽出しました！")
                        st.rerun()
                    else:
                        st.warning("緑色のマーカーが見つからないか、文字が認識できませんでした。")

    with col_right:
        st.subheader("重要単語・意味のリスト")
        st.caption("※単語や意味はタップして修正・書き込みが可能です。")
        
        updated_cards = []
        for idx, card in enumerate(st.session_state.card_data):
            c1, c2 = st.columns(2)
            with c1:
                front = st.text_input(f"単語 {idx+1}", value=card["front"], key=f"f_{idx}")
            with c2:
                back = st.text_input(f"意味 {idx+1}", value=card["back"], key=f"b_{idx}")
            
            updated_cards.append({
                "id": card["id"],
                "front": front,
                "back": back,
                "correct": card["correct"],
                "total": card["total"]
            })
        
        st.session_state.card_data = updated_cards
        
        if st.button("＋ 手動で単語枠を追加", use_container_width=True):
            st.session_state.card_data.append({
                "id": len(st.session_state.card_data) + 1,
                "front": "", "back": "", "correct": 0, "total": 0
            })
            st.rerun()

    st.markdown("---")
    st.subheader("📝 ノート・要約テキスト書き込み（Apple Pencil / 手書き対応）")
    st.session_state.notes = st.text_area(
        "講義メモやテキストをここに入力できます（スクリブル機能対応）", 
        value=st.session_state.notes, 
        height=150
    )

# ==========================================
# 画面②：単語カード学習モード
# ==========================================
elif app_mode == "画面②: 単語カード学習":
    st.header("🎴 画面②: 単語カード学習モード")

    mode = st.radio("学習モード選択", ["すべて学習", "弱点克服モード（正答率 50% 未満）"], horizontal=True)

    cards_to_learn = st.session_state.card_data
    if mode == "弱点克服モード（正答率 50% 未満）":
        cards_to_learn = [
            c for c in st.session_state.card_data 
            if c["total"] > 0 and (c["correct"] / c["total"]) < 0.5
        ]

    if not cards_to_learn:
        st.success("🎉 対象となる単語カードがありません！")
    else:
        if st.session_state.current_index >= len(cards_to_learn):
            st.session_state.current_index = 0

        current_card = cards_to_learn[st.session_state.current_index]

        st.subheader(f"カード {st.session_state.current_index + 1} / {len(cards_to_learn)}")
        
        # カードデザイン（iPadでタップしやすい巨大エリア）
        with st.container(border=True):
            st.markdown(f"<h2 style='text-align: center;'>{current_card['front']}</h2>", unsafe_allow_html=True)
            
            if st.session_state.show_answer:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color: #1E88E5;'>{current_card['back']}</h3>", unsafe_allow_html=True)
            else:
                st.write("")
                if st.button("💡 答 え を 表 示", use_container_width=True):
                    st.session_state.show_answer = True
                    st.rerun()

        # 正解 / 不正解 ボタン
        if st.session_state.show_answer:
            st.write("")
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                if st.button("✕ 不正解", use_container_width=True, type="secondary"):
                    for c in st.session_state.card_data:
                        if c["id"] == current_card["id"]:
                            c["total"] += 1
                    st.session_state.show_answer = False
                    st.session_state.current_index = (st.session_state.current_index + 1) % len(cards_to_learn)
                    st.rerun()

            with btn_col2:
                if st.button("◯ 正解", use_container_width=True, type="primary"):
                    for c in st.session_state.card_data:
                        if c["id"] == current_card["id"]:
                            c["correct"] += 1
                            c["total"] += 1
                    st.session_state.show_answer = False
                    st.session_state.current_index = (st.session_state.current_index + 1) % len(cards_to_learn)
                    st.rerun()

    st.markdown("---")
    st.subheader("📊 弱点単語データの出力")
    weak_words = [
        c for c in st.session_state.card_data 
        if c["total"] > 0 and (c["correct"] / c["total"]) < 0.5
    ]
    
    if weak_words:
        st.warning(f"現在、間違えやすい単語が **{len(weak_words)} 件** あります。")
        df_weak = pd.DataFrame(weak_words)[["front", "back", "correct", "total"]]
        st.dataframe(df_weak, use_container_width=True)
        
        json_string = json.dumps(weak_words, ensure_ascii=False, indent=2)
        st.download_button(
            label="弱点単語リストをJSONでダウンロード",
            data=json_string,
            file_name="weak_words.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.info("現在、弱点判定された単語はありません。")