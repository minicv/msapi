import requests
# If you are using a Jupyter notebook, uncomment the following line.
#%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

import xlrd
import xlwt
from xlutils.copy import copy
from skimage import io
import time

import pdb

# Replace <Subscription Key> with your valid subscription key.
subscription_key = "2539901f75004c7f89254bae875d6a77"
assert subscription_key

# You must use the same region in your REST call as you used to get your
# subscription keys. For example, if you got your subscription keys from
# westus, replace "westcentralus" in the URI below with "westus".
#
# Free trial subscription keys are generated in the westcentralus region.
# If you use a free trial subscription key, you shouldn't need to change
# this region.
vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"

analyze_url = vision_base_url + "analyze"

headers = {'Ocp-Apim-Subscription-Key': subscription_key }
params  = {'visualFeatures': 'Categories,Description,Tags','details':'Celebrities','language':'zh'}


#Open the excel file which is needed to be written
xlsfile = r"/home/mrc/MCcode/minicv/article_data/toutiao_content_image.xlsx"# 打开指定路径中的xls文件
r_article = xlrd.open_workbook(xlsfile)#得到Excel文件的book对象，实例化对象
r_sheet = r_article.sheet_by_index(0)
#w_article = copy(r_article)
#w_sheet = w_article.get_sheet(0) # 通过sheet索引获得sheet对象

new_xl = xlwt.Workbook(encoding='utf-8',style_compression=0)
new_sheet = new_xl.add_sheet('sheet1',cell_overwrite_ok=True)
new_sheet.write(0,0,'article_id')
new_sheet.write(0,1,'images_url')
new_sheet.write(0,2,'tags')
new_sheet.write(0,3,'human_tag')
new_sheet.write(0,4,'captions')
new_xl.save(r"/home/mrc/MCcode/minicv/article_attr_phase.xls")

results_urls = ''
object_tags = ''
human_tag = ''
captions = ''

for xl_row in range(0,r_sheet.nrows):
	url_images = r_sheet.cell_value(xl_row,4)
	if len(url_images) <1:
		continue
	url_images = url_images.split(',')

	if len(url_images)>5:
		url_images = url_images[0:5]
	for url in range(0,len(url_images)):

		image_url = url_images[url]
		image = io.imread(image_url)

		if image.shape[0]<=50 or image.shape[1]<=50:
			continue

		results_urls = results_urls + url_images[url] + ';'

		data    = {'url': image_url}
		response = requests.post(analyze_url, headers=headers, params=params, json=data)
		response.raise_for_status()

		# The 'analysis' object contains various fields that describe the image. The most
		# relevant caption for the image is obtained from the 'description' property.
		analysis = response.json()
		time.sleep(1)

		if len(analysis['categories']) > 0:
			if 'detail' in analysis['categories'][0].keys():
				if 'celebrities' in analysis['categories'][0]['detail'] and len(analysis['categories'][0]['detail']['celebrities'])>0:
					if analysis['categories'][0]['detail']['celebrities'][0]['confidence'] >0.7:
						human_tag = human_tag + analysis['categories'][0]['detail']['celebrities'][0]['name'] + ';'

		if len(analysis['description']) > 0:
			if 'captions' in analysis['description'].keys() and len(analysis['description']['captions'])>0:
				if analysis['description']['captions'][0]['confidence'] > 0.7:
					captions = captions + analysis['description']['captions'][0]['text'] + ';'

		if len(analysis['tags']) > 0:
			for tag_num in range(0,len(analysis['tags'])):
				if analysis['tags'][tag_num]['confidence']>0.7:
					object_tags = object_tags + analysis['tags'][tag_num]['name'] + ','
		if object_tags!='':
			object_tags = object_tags[:-1] + ';'

	new_sheet.write(xl_row+1,0,str(xl_row+1))
	if results_urls!='':
		new_sheet.write(xl_row+1,1,results_urls[:-1])
	new_sheet.write(xl_row+1+575,2,object_tags)
	new_sheet.write(xl_row+1+575,3,human_tag)
	new_sheet.write(xl_row+1+575,4,captions)
	new_xl.save("article_attr_phase.xls")
	results_urls = ''
	object_tags = ''
	human_tag = ''
	captions = ''

'''
image_caption = analysis["description"]["captions"][0]["text"].capitalize()

# Display the image and overlay it with the caption.
image = Image.open(BytesIO(requests.get(image_url).content))
plt.imshow(image)
plt.axis("off")
_ = plt.title(image_caption, size="x-large", y=-0.1)



'''
