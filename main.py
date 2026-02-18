import os
import time
import ccxt
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

# 1. BAĞLANTILAR (Render Panelindeki Environment Variables'dan çekilecek)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MEXC_API_KEY = os.getenv("MEXC_API_KEY")
MEXC_SECRET_KEY = os.getenv("MEXC_SECRET_KEY")

# Ücretsiz AI Modeli (Llama 3 via Groq)
llm = ChatGroq(
    temperature=0.1, 
    model_name="llama3-70b-8192", 
    groq_api_key=GROQ_API_KEY
)

# MEXC Borsasına Bağlantı
exchange = ccxt.mexc({
    'apiKey': MEXC_API_KEY,
    'secret': MEXC_SECRET_KEY,
    'enableRateLimit': True,
})

# 2. ÖZEL FONKSİYON: MEXC'den Veri Çekme
def get_mexc_market_data():
    try:
        # MEXC'deki popüler çiftlerin fiyatlarını çekiyoruz
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'MX/USDT']
        tickers = exchange.fetch_tickers(symbols)
        data_summary = ""
        for symbol, ticker in tickers.items():
            data_summary += f"{symbol}: Fiyat: {ticker['last']}, 24s Değişim: %{ticker['percentage']}\n"
        return data_summary
    except Exception as e:
        return f"Veri çekme hatası: {str(e)}"

# 3. AGENT (TEMSİLCİ) TANIMLARI
analyst = Agent(
    role='MEXC Piyasa Analisti',
    goal='MEXC verilerini inceleyerek kısa vadeli trendleri tespit etmek.',
    backstory='Sen teknik verileri okuma konusunda uzmanlaşmış bir AI agentsın. Sadece verilere odaklanırsın.',
    llm=llm,
    verbose=True
)

risk_manager = Agent(
    role='Risk Yönetim Müdürü',
    goal='Analistin bulgularını denetlemek ve işlem güvenliğini onaylamak.',
    backstory='Senin görevin sermaye kaybını önlemek. Şüpheli piyasa hareketlerinde işlem izni vermezsin.',
    llm=llm,
    verbose=True
)

# 4. İŞ AKIŞI FONKSİYONU
def run_crypto_crew():
    market_info = get_mexc_market_data()
    
    # Görevleri Belirle
    task_analysis = Task(
        description=f"Aşağıdaki canlı MEXC verilerini analiz et ve alım fırsatı var mı bak:\n{market_info}",
        expected_output="Kısa bir teknik analiz özeti ve önerilen coin.",
        agent=analyst
    )

    task_risk = Task(
        description="Analistin önerisini kontrol et. Eğer piyasa çok oynaksa 'BEKLE' kararı ver.",
        expected_output="Final kararı: AL, SAT veya BEKLE.",
        agent=risk_manager
    )

    # Ekibi Kur
    crew = Crew(
        agents=[analyst, risk_manager],
        tasks=[task_analysis, task_risk],
        process=Process.sequential
    )

    return crew.kickoff()

# 5. ANA DÖNGÜ (Sistemi Başlat)
if __name__ == "__main__":
    print("🚀 AI Agent Ekibi Başlatıldı...")
    while True:
        try:
            result = run_crypto_crew()
            print(f"\n--- EKİP RAPORU ---\n{result}\n------------------\n")
            # 30 Dakikada bir çalışması için bekleme süresi
            time.sleep(1800) 
        except Exception as e:
            print(f"⚠️ Kritik Hata: {e}")
            time.sleep(60)

