def pull_to_azure(text):
	import requests, uuid, json

	# Add your key and endpoint
	key = "4WszHyHv10ULCVtQXsJQd2YEJNmYsxwx53XBBrZaU7oyRnIeqxfIJQQJ99BCACYeBjFXJ3w3AAAbACOGXSYp"
	endpoint = "https://api.cognitive.microsofttranslator.com"

	# location, also known as region.
	# required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
	location = "eastus"

	path = '/translate'
	constructed_url = endpoint + path

	params = {
    	'api-version': '3.0',
    	'from': ['en'],
    	'to': ['es'] #languages
	}
	# print(params)
	headers = {
    	'Ocp-Apim-Subscription-Key': key,
    	# location required if you're using a multi-service or regional (not global) resource.
    	'Ocp-Apim-Subscription-Region': location,
    	'Content-type': 'application/json',
    	'X-ClientTraceId': str(uuid.uuid4())
	}

	# You can pass more than one object in body. What you want to translate
	body = [
    	{'text': text['text']}
	]

	request = requests.post(constructed_url, params=params, headers=headers, json=body)
	response = request.json()

	print(json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))
pull_to_azure({'text' : "Hello"})
