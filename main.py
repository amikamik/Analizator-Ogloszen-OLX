import pandas as pd
import openai
import json
import time
import requests
import csv
from datetime import datetime, timezone
from tqdm import tqdm
import re

# Importowanie konfiguracji z osobnego pliku
import config

# ==============================================================================
# ===================   ZUNIFIKOWANA KONFIGURACJA   ====================
# ==============================================================================

# --- Klucze API i Konfiguracja są teraz w pliku config.py ---

# --- Konfiguracja Etapu 1: Skanowanie i Filtracja ---
MIN_REWARD_SCORE = 0.5
MAX_ADS_TO_PROCESS = 500  # Ustaw 0, aby przetworzyć WSZYSTKIE ogłoszenia
REQUESTS_PER_SECOND = 10

# --- Konfiguracja Etapu 2 i 3: Analiza AI ---
MODEL_KATEGORYZACJI = "gpt-4o-mini"
MODEL_EKSPERTA_AUDYTORA = "gpt-4o"
ROZMIAR_PACZKI_DO_ANALIZY_AI = 25

# --- Nazwy Plików ---
PLIK_KATEGORII = 'kategorie.json'
PLIK_WYNIKOWY = 'ostateczna_weryfikacja.csv'

# --- Parametry Oceny Ogłoszeń ("Nagrody") w Etapie 1 ---
WEIGHTS = {
    'views_per_day': 1,
    'phone_views_per_day': 10,
    'observers_per_day': 20,
    'total_messages_per_day': 40
}
MIN_AD_AGE_DAYS = 0

# --- Ustawienia Techniczne ---
BASE_OLX_API_URL = "https://www.olx.pl/api/partner"
DELAY_BETWEEN_REQUESTS = 1 / REQUESTS_PER_SECOND
OLX_HEADERS = {'Authorization': f'Bearer {config.OLX_ACCESS_TOKEN}', 'Version': '2.0'}
OPENAI_CLIENT = openai.OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None

# ==============================================================================
# =======================   FUNKCJE POMOCNICZE   =======================
# ==============================================================================

def wczytaj_kategorie_i_zbuduj_mapy():
    """Wczytuje kategorie i buduje dwie mapy: pełnych ścieżek i zaawansowaną strukturę drzewa."""
    try:
        with open(PLIK_KATEGORII, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            kategorie_lista = json_data['data'] if isinstance(json_data, dict) and 'data' in json_data else json_data

        id_do_nazwy = {kat['id']: kat['name'] for kat in kategorie_lista}
        id_do_rodzica = {kat['id']: kat.get('parent_id') for kat in kategorie_lista if kat.get('parent_id')}
        mapa_sciezek = {}
        for cat_id in id_do_nazwy:
            sciezka, aktualny_id = [id_do_nazwy[cat_id]], cat_id
            while aktualny_id in id_do_rodzica:
                id_rodzica = id_do_rodzica[aktualny_id]
                sciezka.insert(0, id_do_nazwy.get(id_rodzica, '???'))
                aktualny_id = id_rodzica
            mapa_sciezek[cat_id] = " > ".join(sciezka)

        mapa_zaawansowana = {}
        for kat in kategorie_lista:
            if 'id' in kat:
                mapa_zaawansowana[kat['id']] = {'name': kat.get('name', 'Bez nazwy'), 'parent_id': kat.get('parent_id'), 'is_leaf': kat.get('is_leaf', False), 'children_ids': []}
        for kat_id, kat_data in mapa_zaawansowana.items():
            parent_id = kat_data.get('parent_id')
            if parent_id and parent_id in mapa_zaawansowana:
                mapa_zaawansowana[parent_id]['children_ids'].append(kat_id)

        return mapa_sciezek, mapa_zaawansowana
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku '{PLIK_KATEGORII}'. Upewnij się, że znajduje się on w tym samym folderze.")
        return None, None

def get_total_message_count(ad_id):
    total_messages, offset = 0, 0
    while True:
        try:
            time.sleep(DELAY_BETWEEN_REQUESTS)
            response = requests.get(f"{BASE_OLX_API_URL}/threads?advert_id={ad_id}&offset={offset}&limit=50", headers=OLX_HEADERS)
            response.raise_for_status()
            threads_data = response.json().get('data', [])
            if not threads_data: break
            for thread in threads_data: total_messages += thread.get('total_count', 0)
            if len(threads_data) < 50: break
            offset += len(threads_data)
        except requests.exceptions.RequestException: break
    return total_messages

def calculate_reward(stats, total_messages_count, total_age_days):
    if total_age_days < MIN_AD_AGE_DAYS: return 0
    total_age_days = max(1, total_age_days)
    reward = ((stats.get('advert_views', 0) / total_age_days) * WEIGHTS['views_per_day'] +
              (stats.get('phone_views', 0) / total_age_days) * WEIGHTS['phone_views_per_day'] +
              (stats.get('users_observing', 0) / total_age_days) * WEIGHTS['observers_per_day'] +
              (total_messages_count / total_age_days) * WEIGHTS['total_messages_per_day'])
    return reward

def czysc_opis(opis, tytul):
    if not isinstance(opis, str): return ""
    for wzorzec in ["W razie pytań lub wątpliwości", "Specyfikacja:"]:
        if wzorzec in opis: opis = opis.split(wzorzec)[0].strip()
    if opis.startswith(tytul): opis = opis[len(tytul):].strip()
    for wzorzec in ["Towar powystawowy, outletowy, stoki magazynowe.", "Produkt fabrycznie nowy, nieużywany w oryginalnym opakowaniu."]:
        opis = re.sub(wzorzec, '', opis, flags=re.IGNORECASE).strip()
    return opis

def get_sciezke_kategorii(kat_id, mapa_zaawansowana):
    try: kat_id = int(kat_id)
    except (ValueError, TypeError): return str(kat_id)
    if kat_id not in mapa_zaawansowana: return f"Nieznane ID: {kat_id}"
    sciezka, aktualny_id = [], kat_id
    while aktualny_id is not None and aktualny_id != 0:
        if aktualny_id in mapa_zaawansowana:
            sciezka.append(mapa_zaawansowana[aktualny_id]['name'])
            aktualny_id = mapa_zaawansowana[aktualny_id].get('parent_id')
        else: break
    return " > ".join(reversed(sciezka))

# ==============================================================================
# =========================   GŁÓWNE ETAPY PROCESU   =========================
# ==============================================================================

def etap1_skanuj_i_filtruj(mapa_sciezek):
    print("\n" + "="*80)
    print("--- ETAP 1: Skanowanie i Filtracja Ogłoszeń na OLX ---")
    print("="*80)

    high_performing_ads = []
    offset, processed_ads_count, limit_reached = 0, 0, False
    progress_bar_total = MAX_ADS_TO_PROCESS if MAX_ADS_TO_PROCESS > 0 else None
    
    statusy_zakonczone = ['removed_by_user', 'outdated']
    print(f"INFO: Skrypt będzie analizował tylko ogłoszenia o statusach: {', '.join(statusy_zakonczone)}")

    with tqdm(total=progress_bar_total, desc="Skanowanie ogłoszeń", unit=" ogł.") as pbar:
        while not limit_reached:
            time.sleep(DELAY_BETWEEN_REQUESTS)
            try:
                response = requests.get(f"{BASE_OLX_API_URL}/adverts?offset={offset}&limit=50", headers=OLX_HEADERS)
                response.raise_for_status()
                page_of_ads = response.json().get('data', [])
            except requests.exceptions.RequestException as e:
                print(f"\nBłąd API podczas pobierania strony: {e}. Zakończono skanowanie."); break

            if not page_of_ads: break

            for ad_data in page_of_ads:
                if MAX_ADS_TO_PROCESS > 0 and processed_ads_count >= MAX_ADS_TO_PROCESS:
                    limit_reached = True; break

                processed_ads_count += 1
                pbar.update(1)

                ad_id = ad_data.get('id')
                if not ad_id: continue
                
                ad_status = ad_data.get('status')
                if ad_status not in statusy_zakonczone:
                    continue

                try:
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                    stats_res = requests.get(f"{BASE_OLX_API_URL}/adverts/{ad_id}/statistics", headers=OLX_HEADERS)
                    stats_res.raise_for_status()
                    stats = stats_res.json().get('data', {})

                    total_messages_count = get_total_message_count(ad_id)
                    created_at_str = ad_data.get('created_at')
                    total_age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(created_at_str.replace(' ', 'T')).replace(tzinfo=timezone.utc)).days if created_at_str else 0
                    reward = calculate_reward(stats, total_messages_count, total_age_days)

                    if reward > MIN_REWARD_SCORE:
                        high_performing_ads.append({
                            'ID Ogłoszenia': ad_data['id'],
                            'Tytuł': ad_data['title'],
                            'Opis': ad_data['description'],
                            'ID Kategorii': ad_data['category_id'],
                            'Pełna ścieżka kategorii': mapa_sciezek.get(ad_data.get('category_id'), "Brak ścieżki")
                        })
                except (requests.exceptions.RequestException, KeyError, TypeError): continue
            offset += len(page_of_ads)

    print(f"\n--- Zakończono Etap 1 ---")
    print(f"✅ Przeskanowano {processed_ads_count} ogłoszeń. Znaleziono {len(high_performing_ads)} zakończonych ofert o wysokim potencjale.")
    return pd.DataFrame(high_performing_ads)

def etap2_reklasyfikuj_z_audytem(df_dobre_ogloszenia, mapa_zaawansowana):
    print("\n" + "="*80)
    print("--- ETAP 2: Inteligentna Reklasyfikacja z Audytem AI ---")
    print("="*80)

    if df_dobre_ogloszenia.empty:
        print("Brak ogłoszeń do reklasyfikacji. Pomijam etap.")
        return pd.DataFrame()

    df_dobre_ogloszenia['Czysty_opis'] = df_dobre_ogloszenia.apply(lambda row: czysc_opis(row['Opis'], row['Tytuł']), axis=1)

    print("\n[Etap 2.1] Rozpoczynam wstępną kategoryzację...")
    system_prompt_etap1 = "Jesteś nawigatorem kategoryzacji. Zawsze wybieraj najbardziej pasującą opcję z listy. Zejdź jak najgłębiej. Odpowiedz tylko i wyłącznie numerem ID wybranej opcji."
    wyniki_etapu1 = []

    for index, wiersz in tqdm(df_dobre_ogloszenia.iterrows(), total=len(df_dobre_ogloszenia), desc="Kategoryzacja AI"):
        sugestia_ai_id = None
        aktualne_opcje_ids = [kat_id for kat_id, kat in mapa_zaawansowana.items() if kat.get('parent_id') == 0]
        sciezka_wyboru = []
        krok = 1
        while True:
            if not aktualne_opcje_ids: break
            opcje_dla_ai = "\n".join([f"{kat_id}: {mapa_zaawansowana[kat_id]['name']}" for kat_id in aktualne_opcje_ids])
            sciezka_str = " > ".join(sciezka_wyboru)
            user_prompt = f"""Krok {krok}. Aktualna ścieżka: "{sciezka_str if sciezka_str else 'START'}". Wybierz najlepszą podkategorię z listy dla produktu poniżej.\nTytuł: "{wiersz['Tytuł']}"\nOpis: "{wiersz['Czysty_opis']}"\n--- OPCJE ---\n{opcje_dla_ai}\n--- KONIEC ---\nPodaj tylko ID najlepszej opcji:"""
            try:
                response = OPENAI_CLIENT.chat.completions.create(model=MODEL_KATEGORYZACJI, messages=[{"role": "system", "content": system_prompt_etap1}, {"role": "user", "content": user_prompt}], temperature=0.0, timeout=60)
                wybrane_id_str = response.choices[0].message.content.strip()
                if not wybrane_id_str.isdigit() or int(wybrane_id_str) not in aktualne_opcje_ids:
                    sugestia_ai_id = sugestia_ai_id or 'BŁĄD_ETAP1'; break
                wybrane_id = int(wybrane_id_str)
                sugestia_ai_id = wybrane_id
                sciezka_wyboru.append(mapa_zaawansowana[wybrane_id]['name'])
                if mapa_zaawansowana[wybrane_id]['is_leaf'] or not mapa_zaawansowana[wybrane_id]['children_ids']: break
                aktualne_opcje_ids = mapa_zaawansowana[wybrane_id]['children_ids']
                krok += 1
            except Exception as e:
                print(f" -> BŁĄD API w Etapie 1 dla ID {wiersz['ID Ogłoszenia']}: {e}"); sugestia_ai_id = 'BŁĄD_API_1'; break
            time.sleep(0.2)

        wyniki_etapu1.append({'oryginalny_wiersz': wiersz.to_dict(), 'Sugestia AI (Etap 1)': sugestia_ai_id})

    print(f"✅ Zakończono kategoryzację dla {len(wyniki_etapu1)} ogłoszeń.")

    print("\n[Etap 2.2] Przeprowadzam audyt ekspercki wyników...")
    system_prompt_etap2 = """Jesteś Sędzią-Ekspertem. Oceń, czy 'Sugerowana kategoria' jest poprawna i lepsza lub równie dobra jak 'Oryginalna kategoria'. Odpowiedź MUSI być obiektem JSON z kluczem "wyniki_audytu", zawierającym listę obiektów: {"id_ogloszenia": int, "ocena": "dobra"|"zła", "komentarz": "..."}."""
    audyt_mapa = {}
    for i in tqdm(range(0, len(wyniki_etapu1), ROZMIAR_PACZKI_DO_ANALIZY_AI), desc="Audyt Ekspercki AI"):
        paczka = wyniki_etapu1[i:i + ROZMIAR_PACZKI_DO_ANALIZY_AI]
        dane_do_audytu_str = ""
        for wynik in paczka:
            wiersz = wynik['oryginalny_wiersz']
            dane_do_audytu_str += f"---\nID Ogłoszenia: {wiersz['ID Ogłoszenia']}\nTytuł: {wiersz['Tytuł']}\nOpis: {wiersz['Czysty_opis']}\nOryginalna kategoria: {wiersz['Pełna ścieżka kategorii']}\nSugerowana kategoria: {get_sciezke_kategorii(wynik['Sugestia AI (Etap 1)'], mapa_zaawansowana)}\n"
        user_prompt = f"Oceń poniższe wyniki kategoryzacji i zwróć listę JSON w wymaganym formacie:\n\n{dane_do_audytu_str}"
        try:
            response = OPENAI_CLIENT.chat.completions.create(model=MODEL_EKSPERTA_AUDYTORA, response_format={"type": "json_object"}, messages=[{"role": "system", "content": system_prompt_etap2}, {"role": "user", "content": user_prompt}], temperature=0.0, timeout=400)
            audyt_dane = json.loads(response.choices[0].message.content.strip())
            for item in audyt_dane.get('wyniki_audytu', []):
                if isinstance(item, dict) and all(k in item for k in ['id_ogloszenia', 'ocena', 'komentarz']):
                    audyt_mapa[item['id_ogloszenia']] = {'ocena': item['ocena'], 'komentarz': item['komentarz']}
        except Exception as e:
            print(f" -> KRYTYCZNY BŁĄD podczas audytu paczki: {e}")
        time.sleep(1)

    print("\n[Etap 2.3] Koryguję błędne sugestie na podstawie audytu...")
    system_prompt_etap3 = "Jesteś inteligentnym asystentem. Wybierz LEPSZĄ kategorię z dwóch opcji, biorąc pod uwagę komentarz eksperta. Odpowiedz tylko i wyłącznie numerem ID wybranej kategorii."
    finalne_wyniki_korekty = []
    for wynik in tqdm(wyniki_etapu1, desc="Korekta po audycie"):
        wiersz = wynik['oryginalny_wiersz']
        id_ogloszenia = wiersz['ID Ogłoszenia']
        audyt = audyt_mapa.get(id_ogloszenia)
        finalny_id = wynik['Sugestia AI (Etap 1)']

        if audyt and audyt['ocena'] == 'zła':
            user_prompt = f"""Produkt: "{wiersz['Tytuł']}"\nOpis: "{wiersz['Czysty_opis']}"\nKomentarz eksperta: "{audyt['komentarz']}"\nWybierz lepszą opcję z poniższych:\nOpcja A: {wiersz['ID Kategorii']}: {wiersz['Pełna ścieżka kategorii']}\nOpcja B: {wynik['Sugestia AI (Etap 1)']}: {get_sciezke_kategorii(wynik['Sugestia AI (Etap 1)'], mapa_zaawansowana)}\nPodaj tylko ID lepszej kategorii:"""
            try:
                response = OPENAI_CLIENT.chat.completions.create(model=MODEL_KATEGORYZACJI, messages=[{"role": "system", "content": system_prompt_etap3}, {"role": "user", "content": user_prompt}], temperature=0.0, timeout=60)
                werdykt_str = response.choices[0].message.content.strip()
                if werdykt_str.isdigit() and int(werdykt_str) in [wiersz['ID Kategorii'], wynik['Sugestia AI (Etap 1)']]:
                    finalny_id = int(werdykt_str)
                else: finalny_id = 'BŁĄD_KOREKTY'
            except Exception: finalny_id = 'BŁĄD_API_3'
            time.sleep(0.2)

        wiersz['Sugerowane ID nowej kategorii'] = finalny_id
        finalne_wyniki_korekty.append(wiersz)

    print("\n--- Zakończono Etap 2 ---")
    print(f"✅ Pomyślnie reklasyfikowano {len(finalne_wyniki_korekty)} ogłoszeń.")
    return pd.DataFrame(finalne_wyniki_korekty)

def etap3_ostateczna_weryfikacja(df_reklasyfikowane, mapa_zaawansowana):
    print("\n" + "="*80)
    print("--- ETAP 3: Ostateczna Weryfikacja Jakości przez AI ---")
    print("="*80)

    if df_reklasyfikowane.empty:
        print("Brak ogłoszeń do weryfikacji. Pomijam etap.")
        return pd.DataFrame()

    df_reklasyfikowane['Sugerowana pełna ścieżka'] = df_reklasyfikowane['Sugerowane ID nowej kategorii'].apply(lambda x: get_sciezke_kategorii(x, mapa_zaawansowana))

    system_prompt_audytora = """Jesteś ostatecznym audytorem jakości. Oceń poprawność przypisanej kategorii. Zwróć ocenę pewności w skali 1-5 (5=idealna, 1=błąd). Odpowiedź MUSI być obiektem JSON z kluczem "wyniki_audytu", zawierającym listę obiektów: {"id_ogloszenia": int, "ocena_pewnosci": int, "uzasadnienie": "..."}."""

    audyt_mapa = {}
    for i in tqdm(range(0, len(df_reklasyfikowane), ROZMIAR_PACZKI_DO_ANALIZY_AI), desc="Finalna weryfikacja AI"):
        paczka = df_reklasyfikowane.iloc[i:i + ROZMIAR_PACZKI_DO_ANALIZY_AI]
        dane_do_audytu_str = ""
        for index, row in paczka.iterrows():
            dane_do_audytu_str += f"---\nID Ogłoszenia: {row['ID Ogłoszenia']}\nTytuł: {row['Tytuł']}\nOpis: {row['Czysty_opis']}\nOSTATECZNA KATEGORIA: {row['Sugerowana pełna ścieżka']}\n"
        user_prompt = f"Oceń poniższe wyniki kategoryzacji i zwróć listę JSON w wymaganym formacie:\n\n{dane_do_audytu_str}"
        try:
            response = OPENAI_CLIENT.chat.completions.create(model=MODEL_EKSPERTA_AUDYTORA, response_format={"type": "json_object"}, messages=[{"role": "system", "content": system_prompt_audytora}, {"role": "user", "content": user_prompt}], temperature=0.0, timeout=400)
            audyt_dane = json.loads(response.choices[0].message.content.strip())
            for item in audyt_dane.get('wyniki_audytu', []):
                if isinstance(item, dict) and all(k in item for k in ['id_ogloszenia', 'ocena_pewnosci', 'uzasadnienie']):
                    audyt_mapa[item['id_ogloszenia']] = {'Ocena_Pewnosci': item['ocena_pewnosci'], 'Uzasadnienie_Audytora': item['uzasadnienie']}
        except Exception as e:
            print(f" -> KRYTYCZNY BŁĄD podczas audytu paczki: {e}")
        time.sleep(1)

    df_audytu = pd.DataFrame.from_dict(audyt_mapa, orient='index')
    df_audytu.index.name = 'ID Ogłoszenia'
    df_wynikowe = df_reklasyfikowane.merge(df_audytu, on='ID Ogłoszenia', how='left')
    df_wynikowe['Ocena_Pewnosci'].fillna(0, inplace=True)
    df_wynikowe['Uzasadnienie_Audytora'].fillna('Audyt nie powiódł się', inplace=True)

    df_wynikowe.to_csv(PLIK_WYNIKOWY, index=False, sep=';', encoding='utf-8-sig')

    print("\n--- Zakończono Etap 3 ---")
    print(f"✅ Pomyślnie zweryfikowano {len(df_wynikowe)} ogłoszeń. Wyniki zapisano do '{PLIK_WYNIKOWY}'.")
    return df_wynikowe

# ==============================================================================
# ===================   GŁÓWNA FUNKCJA URUCHOMIENIOWA   ==================
# ==============================================================================

def main():
    """Orkiestruje cały proces od A do Z."""
    print("#"*80)
    print("#####   START ZUNIFIKOWANEGO PROCESU ANALIZY OGŁOSZEŃ OLX   #####")
    print("#"*80)

    if not config.OPENAI_API_KEY or not config.OLX_ACCESS_TOKEN:
        print("\n❌ BŁĄD KRYTYCZNY: Brak kluczy API w pliku config.py. Zakończono.")
        return
    print("\n✅ Klucze API zostały pomyślnie wczytane.")

    mapa_sciezek, mapa_zaawansowana = wczytaj_kategorie_i_zbuduj_mapy()
    if mapa_sciezek is None:
        return
    print(f"✅ Pomyślnie wczytano i przetworzono {len(mapa_zaawansowana)} kategorii.")

    # Uruchomienie kolejnych etapów
    df_etap1 = etap1_skanuj_i_filtruj(mapa_sciezek)
    df_etap2 = etap2_reklasyfikuj_z_audytem(df_etap1, mapa_zaawansowana)
    df_finalny = etap3_ostateczna_weryfikacja(df_etap2, mapa_zaawansowana)

    print("\n" + "#"*80)
    print("#####   PROCES ZAKOŃCZONY   #####")
    print(f"Końcowy raport został zapisany w pliku: {PLIK_WYNIKOWY}")
    if df_finalny is not None and not df_finalny.empty:
        print("\nPróbka ostatecznych wyników:")
        print(df_finalny[['ID Ogłoszenia', 'Tytuł', 'Sugerowana pełna ścieżka', 'Ocena_Pewnosci']].head())
    print("#"*80)

if __name__ == "__main__":
    main()