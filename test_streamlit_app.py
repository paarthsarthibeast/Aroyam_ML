import time
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By

@pytest.fixture
def driver():
    # Set up Chrome WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode for CI
    driver = webdriver.Chrome(options=options)
    driver.get("http://localhost:8501")  # Assuming Streamlit runs locally on this port
    yield driver
    driver.quit()

def test_streamlit_app(driver):
    # Give the app time to load
    time.sleep(2)
    
    # Interact with a Streamlit widget, for example a button
    button = driver.find_element(By.XPATH, '//button[text()="Click me"]')  # Update with the actual button text
    button.click()
    
    # Assert some UI change happens after clicking
    result_text = driver.find_element(By.XPATH, '//div[@class="stTextElement"]')  # Adjust for your app's structure
    assert "Expected Result" in result_text.text
