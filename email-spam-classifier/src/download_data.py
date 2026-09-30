import os
import urllib.request
import zipfile
import io

def download_data():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'spam.csv')

    # Primary: Raw GitHub CSV url
    github_url = "https://raw.githubusercontent.com/mohitgupta-1O1/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
    # Secondary fallback: UCI ZIP url
    uci_zip_url = "https://archive.ics.ucs.edu/static/public/228/sms+spam+collection.zip"

    print("Attempting to download dataset from raw GitHub repository...")
    try:
        urllib.request.urlretrieve(github_url, csv_path)
        if os.path.getsize(csv_path) > 100000:
            print(f"Dataset successfully downloaded and saved to {csv_path}")
            return
        else:
            print("Downloaded file size is suspiciously small. Trying fallback UCI link...")
    except Exception as e:
        print(f"Failed to download from GitHub: {e}. Trying fallback UCI link...")

    try:
        print("Attempting to download zip from UCI ML Repository...")
        with urllib.request.urlopen(uci_zip_url) as response:
            zip_data = response.read()
        
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            file_list = z.namelist()
            print(f"Zip files: {file_list}")
            target_file = None
            for name in ['SMSSpamCollection', 'smsspamcollection', 'SMSSpamCollection.txt']:
                if name in file_list:
                    target_file = name
                    break
            
            if not target_file:
                target_file = file_list[0]
            
            print(f"Extracting {target_file}...")
            content = z.read(target_file).decode('latin-1')
            
            import csv
            lines = content.strip().split('\n')
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['v1', 'v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
                for line in lines:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        writer.writerow([parts[0], parts[1], '', '', ''])
                    elif len(parts) == 1:
                        writer.writerow([parts[0], '', '', '', ''])
            
            print(f"Dataset successfully processed and saved to {csv_path}")
    except Exception as e:
        print(f"Failed to download from UCI: {e}")
        raise RuntimeError("Could not acquire SMS Spam Collection dataset.")

if __name__ == '__main__':
    download_data()
