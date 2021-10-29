import requests

request=requests.get('https://api.coindesk.com/v1/bpi/currentprice.json')

#print(request.status_code)

#print(request.text)

data=request.json()
# Coindesk API: https://www.coindesk.com/api/
print('The price of Bitcoin is currently: ' + data['time']['updated'], data['chartName'], data['bpi']['USD']['rate'])

