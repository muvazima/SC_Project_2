import requests
#username='mansoorm'
#username='makarana'
username='bonus'
url='http://cs7ns1.scss.tcd.ie/2122/'+username+'/pi-project2-bonus/'
r=requests.get(url+username+'-challenge-filenames.csv')
for i in r.text.split(',\n'):
   captcha=requests.get(url+i)
   file_loc=username+'_captchas/'+i
   file=open(file_loc, 'wb')
   file.write(captcha.content)
   file.close()
