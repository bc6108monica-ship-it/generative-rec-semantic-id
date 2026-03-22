import requests
import time

# 换成你的中转 API 地址
url = "https://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/v1/messages" 

while True:
    try:
        response = requests.get(url)
        if response.status_code != 503:
            print(f"[{time.strftime('%H:%M:%S')}] 状态更新！当前状态码: {response.status_code}，可能已恢复！")
            break
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 还在维护中...")
    except Exception as e:
        print("连接异常，等待重试...")
    
    time.sleep(60) # 每分钟检测一次