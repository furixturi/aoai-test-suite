import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

# Azure OpenAI API settings

# API_ENDPOINT = os.environ["PAYGO_GPT_35_T_JP_ENDPOINT"]
# DEPLOYMENT = os.environ["PAYGO_GPT_35_T_DEPLOYMENT"]
# API_KEY = os.environ["PAYGO_GPT_35_T_JP_KEY"]
# API_ENDPOINT = os.environ["PAYGO_GPT_4_AU_ENDPOINT"]
# DEPLOYMENT = os.environ["PAYGO_GPT_4_AU_DEPLOYMENT"]
# API_KEY = os.environ["PAYGO_GPT_4_AU_KEY"]


# API_ENDPOINT = os.environ["PTU_4o_SWC_ENDPOINT"]
# DEPLOYMENT = os.environ["PTU_4o_SWC_DEPLOYMENT"]
# API_KEY = os.environ["PTU_4o_SWC_KEY"]

# API_ENDPOINT = os.environ["PAYGO_4o_SWC_ENDPOINT"]
# DEPLOYMENT = os.environ["PAYGO_4o_SWC_DEPLOYMENT"]
# API_KEY = os.environ["PAYGO_4o_SWC_KEY"]

## 4o PayGo Standard
# API_ENDPOINT = os.environ["PAYGO_4o_ENDPOINT"]
# DEPLOYMENT = os.environ["PAYGO_4o_DEPLOYMENT"]
# API_KEY = os.environ["PAYGO_4o_KEY"]

# 2024/08/13
# API_ENDPOINT = os.environ["PTU_4o_SWC_2_ENDPOINT"]
# DEPLOYMENT = os.environ["PTU_4o_SWC_2_DEPLOYMENT"]
# API_KEY = os.environ["PTU_4o_SWC_2_KEY"]

# 2024/09/13
API_ENDPOINT = os.environ["GPT_4o_SWC_20240913_ENDPOINT"]
DEPLOYMENT = os.environ["GPT_4o_SWC_20240913_PTU_DEPLOYMENT"]
API_KEY = os.environ["GPT_4o_SWC_20240913_KEY"]

# Azure OpenAI API settings
API_ENDPOINT = f"{API_ENDPOINT}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version=2024-06-01"
HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY
}

# Define a long prompt in Japanese (approx. 7500 tokens) for a banking use case
LONG_PROMPT = (
    "銀行業務において、お客様に最高のサービスを提供するためには、様々な手続きやプロセスが必要です。"
    "ここでは、一般的な銀行業務のフローについて詳しく説明します。"
    "まず、お客様が新しい口座を開設する際には、個人情報の確認と本人確認書類の提出が必要です。"
    "口座開設手続きには、次のステップが含まれます："
    "1. 申込書の記入：お客様は銀行が提供する申込書に必要事項を記入します。"
    "2. 本人確認書類の提出：お客様は身分証明書（運転免許証、パスポートなど）を提出し、銀行が本人確認を行います。"
    "3. 初回入金：口座開設後、初回の入金が必要です。お客様は現金または振込で初回入金を行います。"
    "口座開設後、お客様はインターネットバンキングやモバイルバンキングの登録を行い、オンラインでの取引を開始することができます。"
    "インターネットバンキングの登録手順には次のステップが含まれます："
    "1. ウェブサイトへのアクセス：お客様は銀行のウェブサイトにアクセスします。"
    "2. ユーザーIDとパスワードの設定：お客様は自分のユーザーIDとパスワードを設定します。"
    "3. 認証コードの受信と入力：お客様は銀行から送られる認証コードを受け取り、それをウェブサイトに入力します。"
    "次に、銀行はお客様の資産を安全に管理し、預金、送金、投資、ローンなどの各種サービスを提供します。"
    "特に、ローンの申請プロセスは詳細な審査が必要です。お客様の信用情報、収入、既存の債務などを総合的に評価し、適切なローンの金額と金利を決定します。"
    "ローン申請の流れには次のステップが含まれます："
    "1. 事前審査の申込：お客様はローンの事前審査を申し込みます。"
    "2. 必要書類の提出：お客様は収入証明書や納税証明書など、必要な書類を提出します。"
    "3. 本審査：銀行はお客様の書類をもとに本審査を行います。"
    "4. 契約書の締結：審査が通った後、お客様と銀行はローン契約書を締結します。"
    "5. 資金の受領：契約が締結されると、お客様は資金を受け取ります。"
    "投資商品の提供においては、お客様のリスク許容度や投資目標に基づいたアドバイスを行います。"
    "投資商品の選定と購入プロセスには次のステップが含まれます："
    "1. お客様のリスクプロファイルの作成：お客様のリスク許容度を評価します。"
    "2. 適切な投資商品の提案：お客様のプロファイルに基づき、適切な投資商品を提案します。"
    "3. 契約書の確認と署名：お客様は投資商品に関する契約書を確認し、署名します。"
    "4. 投資商品の購入：お客様は提案された投資商品を購入します。"
    "トラブル対応に関しては、お客様が不正取引や口座の問題に直面した場合、速やかにカスタマーサポートに連絡し、適切な対応を受けることが重要です。"
    "銀行はお客様の資産を保護し、安心して取引を行える環境を提供するために、最新のセキュリティ技術を導入しています。"
    "お客様は、定期的に取引明細を確認し、不審な取引がないかチェックすることが推奨されます。"
    "また、パスワードの管理や二要素認証の設定など、セキュリティ強化のための措置を講じることも重要です。"
    "以下に、各種サービスの詳細とその利用方法についてさらに詳しく説明します。"
    "1. 預金サービス：普通預金、定期預金、外貨預金などの各種預金サービスについて。"
    "2. 送金サービス：国内送金、国際送金、リアルタイム送金などの方法と手数料について。"
    "3. ローンサービス：住宅ローン、自動車ローン、教育ローンなどの種類と申請手続きについて。"
    "4. 投資サービス：投資信託、株式、債券などの投資商品の選び方と運用方法について。"
    "5. カスタマーサポート：問い合わせ方法、FAQ、トラブルシューティングガイドなど。"
    "お客様のニーズに応じたパーソナライズドサービスも提供しており、専門のアドバイザーが最適な金融プランを提案します。"
    "最新の金融技術を活用したデジタルバンキングサービスにより、お客様はどこにいても簡単に銀行取引を行うことができます。"
    "セキュリティ面でも、マルチレイヤーの保護システムを導入し、サイバー攻撃や不正アクセスからお客様の資産を守ります。"
    "銀行業務の未来に向けて、継続的にサービスの改善と技術革新を進め、お客様に最先端の金融ソリューションを提供することを目指しています。"
    "お客様との信頼関係を築くために、透明性の高い運用と親切な対応を心がけています。"
    "さらに、社会貢献活動や環境保護への取り組みも行い、地域社会と共に発展することを目指しています。"
    "このように、銀行はお客様に対して包括的な金融サービスを提供し、安全で信頼できる取引環境を提供しています。"
    "ご利用いただくすべてのサービスにおいて、お客様の満足を最優先に考え、日々努力を重ねています。"
    "これからも、お客様の期待に応えるべく、質の高いサービスとサポートを提供してまいります。"
) * 2  # Adjusted to fit the 4500 input token per req 

# 185 output token per req
prompt = LONG_PROMPT + "以上の文章を参考にして、185文字で銀行業務について説明してください。"

generate_tokens = [500, 1000, 2000, 3000, 4000]

generate_token_length = 500

prompt = f"銀行について{generate_token_length}トークンの文章を生成してください。それ以外に何も出リュクしないこと。"

PAYLOAD = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Answer in the user's language."},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": generate_token_length,
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0,
    "presence_penalty": 0
}

def make_request():
    print("Making request at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    try:
        response = requests.post(API_ENDPOINT, headers=HEADERS, json=PAYLOAD)
        if response.status_code == 200:
            print("Success:", response.json())
        else:
            print("Error:", response.status_code, response.text)
    except Exception as e:
        print("Exception occurred:", e)

# Number of requests per minute
# RPM = 480
# RPM = 240 # test env max, half of the test target, assume we use 2 subscriptions and this is one of them
RPM = 20 # upper limit for 4o 50PTU in this request/response pattern

# Calculate the delay between requests in seconds
DELAY = 60 / RPM

RUN_TEST_DURATION = 60 * 5  # 5 minutes with 20 RPM


def test_wait_for_response():
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=RPM // 60) as executor:
        while True:
            # Check if the total runtime has exceeded the specified duration
            if time.time() - start_time > RUN_TEST_DURATION:
                break

            cycle_start_time = time.time()
            # Launch requests
            futures = [executor.submit(make_request) for _ in range(RPM // 60)]
            # Wait for all requests to complete
            for future in futures:
                future.result()

            # Calculate elapsed time for the current cycle
            cycle_elapsed_time = time.time() - cycle_start_time
            # Sleep for the remaining time of the minute if needed
            if cycle_elapsed_time < 60:
                time.sleep(60 - cycle_elapsed_time)

def test_spread_workers():
    print("test_spread_workers")
    start_time = time.time()
    end_time = start_time + RUN_TEST_DURATION
    # futures = []
    with ThreadPoolExecutor(max_workers=RPM) as executor:
        while time.time() < end_time:
            for _ in range(RPM // 60):
                # future = executor.submit(make_request)
                executor.submit(make_request)
                # futures.append(future)
                time.sleep(DELAY)
                
def test_full_concurrency():
    start_time = time.time()
    end_time = start_time + RUN_TEST_DURATION

    with ThreadPoolExecutor(max_workers=RPM) as executor:
        while time.time() < end_time:
            cycle_start_time = time.time()
            # Launch requests for this cycle
            for _ in range(RPM):
                executor.submit(make_request)
                time.sleep(DELAY)  # Spread requests across the minute
            
            # Calculate elapsed time for the current cycle
            cycle_elapsed_time = time.time() - cycle_start_time
            # Adjust for any delay to maintain the RPM
            if cycle_elapsed_time < 60:
                time.sleep(60 - cycle_elapsed_time)




def main():
    print(f" === Test started at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} === ")
    # test_wait_for_response()
    test_spread_workers()
    # test_full_concurrency()

if __name__ == "__main__":
    main()
