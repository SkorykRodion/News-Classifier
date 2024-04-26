from selenium.webdriver.common.by import By
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import datetime
import pandas as pd
# отримання посилання на архівну сторінку за датою
def get_url_by_date_TSN(d):
    url = 'https://tsn.ua/news?day=' + '{:02d}'.format(d.day)
    url+= '&month='+ '{:02d}'.format(d.month) + '&year='+ str(d.year)
    return url

def get_url_by_date_Unian(d):
    url = 'https://www.unian.ua/news/archive/' + str(d.year)
    url += '{:02d}'.format(d.month) + '{:02d}'.format(d.day)
    return url
# парсинг окремої сторінки новин
def parse_TSN_page(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    # перехід в повноекранний режим
    driver.maximize_window()
    # перехід за посиланням
    driver.get(url)
    # очікування завантаження сторінки
    driver.implicitly_wait(15)

    #отримуємо текст з заголовків
    data = []
    flag = True
    while flag:
        headlines = driver.find_elements(By.XPATH, "//a[@class='c-card__link']")
        try:
            # реалізація кроулінгу
            next_el = driver.find_element(By.XPATH,
                                          '''//a[@class='i-before i-arrow-ltr i-arrow--sm']''')
            for headline in headlines:
                if (headline.text != ''):
                    data.append(headline.text)
            driver.execute_script("arguments[0].click();", next_el)
            time.sleep(2)
        except:
            for headline in headlines:
                data.append(headline.text)
            flag = False

    driver.quit()
    return data

def parse_Unian_page(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    # перехід в повноекранний режим
    driver.maximize_window()
    # перехід за посиланням
    driver.get(url)
    # очікування завантаження сторінки
    driver.implicitly_wait(15)

    data = []

    #отримуємо текст з заголовків
    headlines = driver.find_elements(By.XPATH, "//a[@class='list-thumbs__title']")

    for headline in headlines:
        if (headline.text != ''):
            data.append(headline.text)
    driver.quit()
    return data

# створення листа з дати і вказаної кількості днів до неї
def createDateList(base, num):
    date_list = [base - datetime.timedelta(days=x) for x in range(num)]
    return date_list

def parse_news(file = 'news_data.json', base = datetime.datetime.today(), num = 5):
    tsn_list = []
    unian_list = []
    date_list = createDateList(base, num)
    for date in date_list:
        url_tsn = get_url_by_date_TSN(date)
        url_unian = get_url_by_date_Unian(date)
        tmp_tsn = parse_TSN_page(url_tsn)
        print('Date:', date, ' TSN parsed')
        tmp_unian = parse_Unian_page(url_unian)
        print('Date:', date, ' Unian parsed')
        tsn_list.append(tmp_tsn)
        unian_list.append(tmp_unian)
    data = pd.DataFrame()
    data['TSN'] = pd.Series(tsn_list)
    data['Unian'] = pd.Series(unian_list)
    data.index = date_list
    data.to_json(file)

def normalize_json(df):
    for row in df.index:
        for colum in df.columns:
            df.loc[row, colum] = df.loc[row, colum][2:-2].split("', \'")
    return df

def load_data_json(File_name):
    tmp = pd.read_json(File_name)
    return normalize_json(tmp)




