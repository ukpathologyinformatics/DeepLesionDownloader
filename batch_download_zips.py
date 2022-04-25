# Download the 56 zip files in Images_png in batches
import urllib.request
import zipfile
import os
import pandas as pd
import cv2
import csv
import numpy as np
import hashlib
import shutil

###########################
# Create root folder 
# Create 'labels' folder inside of root folder
# Create 'images' folder inside of root folder
##	Create 'test', 'val', 'train' folders inside images folder
# Copy DL_info to root folder (if not already there)
# Pip install pandas, cv2, zipfile, numpy
# Change paths below
# Run it, and pray
###########################

OUTPUT_PATH = 'Y:\\DeepLesion\\' # keep trailing slashes
LABEL_PATH = 'Y:\\DeepLesion\\labels\\'
BORDER_BOX_COORDS_PATH = 'Y:\\DeepLesion\\DL_info.csv'
coords_df = pd.read_csv(BORDER_BOX_COORDS_PATH)



def verify_md5(filename):
	with open(OUTPUT_PATH+"MD5_checksums.txt", 'r') as f:
		row = f.readline()
		if filename in row:
			row_s = row.split("  ")
			check = row_s[0]
			file = row_s[1]
			calc = hashlib.md5(open(OUTPUT_PATH+filename,'rb').read()).hexdigest()
			if calc == check:
				print(filename + ": Checksum OK (" + calc + ":" + check + ")")
				return True
			else:
				print(filename + ": Checksum NOT OK (" + calc + ":" + check + ")")
				return False
	print(filename + " not in MD5 file... skipping.")
	return True


# get all image files in dir
def getListOfFiles(path):
	listOfFile = os.listdir(path)
	allFiles = list()
	# Iterate over all the entries
	for entry in listOfFile:
		# Create full path
		fullPath = os.path.join(path, entry)
		# If entry is a directory then get the list of files in this directory 
		if os.path.isdir(fullPath):
			allFiles = allFiles + getListOfFiles(fullPath)
		else:
			allFiles.append(fullPath)
	return allFiles

# border box coord converter
def convert(x1, y1, x2, y2, image_width, image_height): #may need to normalize
	dw = 1./image_width
	dh = 1./image_height
	x = (x1 + x2)/2.0
	y = (y1 + y2)/2.0
	w = x2 - x1
	h = y2 - y1
	x = x*dw
	w = w*dw
	y = y*dh
	h = h*dh
	return x,y,w,h

def read_DL_info():
	"""read spacings and image indices in DeepLesion"""
	spacings = []
	idxs = []
	with open(BORDER_BOX_COORDS_PATH, 'r') as csvfile:
		reader = csv.reader(csvfile)
		rownum = 0
		for row in reader:
			if rownum == 0:
				header = row
				rownum += 1
			else:
				idxs.append([int(d) for d in row[1:4]])
				spacings.append([float(d) for d in row[12].split(',')])

	idxs = np.array(idxs)
	spacings = np.array(spacings)
	return idxs, spacings

def fix_save_images(src_path):
	images = getListOfFiles(src_path)
	idxs, spacings = read_DL_info()
	train_images = []
	val_images = []
	test_images = []

	print("Moving images and finding bounding box coords...")
	for im in images:
		## move
		image_name_split = im.split('\\')
		image_name = image_name_split[-2] + "_" + image_name_split[-1]
		#curr_image = cv2.imread(im, -1)
		#converted_im = convert_to_png([np.array((curr_image.astype(np.int32) - 32768).astype(np.int16), np.int16)])
		#imageio.imwrite(DST_IMAGE_PATH + image_name, converted_im) # DONT CHANGE THIS IS UINT8
		

		## find coords
		name = coords_df.loc[coords_df['File_name'].str.contains(image_name, case=False)]
		if name.index.size > 0:
			coords = coords_df["Bounding_boxes"][name.index[0]].split(',')
			label = coords_df["Coarse_lesion_type"][name.index[0]]
			train_val_test = coords_df["Train_Val_Test"][name.index[0]]

			if label != -1:     # -1 labels don't mean anything
				#Train
				if train_val_test == 1:     
					train_images.append(OUTPUT_PATH + "images\\train\\" + image_name)
					shutil.move(im, OUTPUT_PATH + "images\\train\\" + image_name)
					with open(LABEL_PATH + image_name[:-3] + "txt", 'a') as f:
						x,y,w,h = convert(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]), 512, 512)
						f.write(str(label-1) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")

				#Validation
				elif train_val_test == 2:   
					val_images.append(OUTPUT_PATH + "images\\val\\" + image_name)
					shutil.move(im, OUTPUT_PATH + "images\\val\\" + image_name)
					with open(LABEL_PATH + image_name[:-3] + "txt", 'a') as f:
						x,y,w,h = convert(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]), 512, 512)
						f.write(str(label-1) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")

				#Test
				elif train_val_test == 3:
					test_images.append(OUTPUT_PATH + "images\\test\\" + image_name)
					shutil.move(im, OUTPUT_PATH + "images\\test\\" + image_name)
					with open(LABEL_PATH + image_name[:-3] + "txt", 'a') as f:
						x,y,w,h = convert(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]), 512, 512)
						f.write(str(label-1) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")

	# remove folder with unused images
	shutil.rmtree(OUTPUT_PATH + 'Images_png\\')
	print("Moving images and finding bounding box coords... done.")

	print("Writing training image names to file...")
	with open(OUTPUT_PATH + 'train.txt', 'a') as f:
		for im in train_images:
			f.write(im + '\n')
	print("Writing training image names to file... done.")

	print("Writing validation image names to file...")
	with open(OUTPUT_PATH + 'val.txt', 'a') as f:
		for im in val_images:
			f.write(im + '\n')
	print("Writing validation image names to file... done.")

	print("Writing testing image names to file...")
	with open(OUTPUT_PATH + 'test.txt', 'a') as f:
		for im in test_images:
			f.write(im + '\n')
	print("Writing testing image names to file... done.")
	


# URLs for the zip files
links = [
	'https://nihcc.box.com/shared/static/sp5y2k799v4x1x77f7w1aqp26uyfq7qz.zip',
	'https://nihcc.box.com/shared/static/l9e1ys5e48qq8s409ua3uv6uwuko0y5c.zip',
	'https://nihcc.box.com/shared/static/48jotosvbrw0rlke4u88tzadmabcp72r.zip',
	'https://nihcc.box.com/shared/static/xa3rjr6nzej6yfgzj9z6hf97ljpq1wkm.zip',
	'https://nihcc.box.com/shared/static/58ix4lxaadjxvjzq4am5ehpzhdvzl7os.zip',
	'https://nihcc.box.com/shared/static/cfouy1al16n0linxqt504n3macomhdj8.zip',
	'https://nihcc.box.com/shared/static/z84jjstqfrhhlr7jikwsvcdutl7jnk78.zip',
	'https://nihcc.box.com/shared/static/6viu9bqirhjjz34xhd1nttcqurez8654.zip',
	'https://nihcc.box.com/shared/static/9ii2xb6z7869khz9xxrwcx1393a05610.zip',
	'https://nihcc.box.com/shared/static/2c7y53eees3a3vdls5preayjaf0mc3bn.zip',

	'https://nihcc.box.com/shared/static/2zsqpzru46wsp0f99eaag5yiad42iezz.zip',
	'https://nihcc.box.com/shared/static/8v8kfhgyngceiu6cr4sq1o8yftu8162m.zip',
	'https://nihcc.box.com/shared/static/jl8ic5cq84e1ijy6z8h52mhnzfqj36q6.zip',
	'https://nihcc.box.com/shared/static/un990ghdh14hp0k7zm8m4qkqrbc0qfu5.zip',
	'https://nihcc.box.com/shared/static/kxvbvri827o1ssl7l4ji1fngfe0pbt4p.zip',
	'https://nihcc.box.com/shared/static/h1jhw1bee3c08pgk537j02q6ue2brxmb.zip',
	'https://nihcc.box.com/shared/static/78hamrdfzjzevrxqfr95h1jqzdqndi19.zip',
	'https://nihcc.box.com/shared/static/kca6qlkgejyxtsgjgvyoku3z745wbgkc.zip',
	'https://nihcc.box.com/shared/static/e8yrtq31g0d8yhjrl6kjplffbsxoc5aw.zip',
	'https://nihcc.box.com/shared/static/vomu8feie1qembrsfy2yaq36cimvymj8.zip',

	'https://nihcc.box.com/shared/static/ecwyyx47p2jd621wt5c5tc92dselz9nx.zip',
	'https://nihcc.box.com/shared/static/fbnafa8rj00y0b5tq05wld0vbgvxnbpe.zip',
	'https://nihcc.box.com/shared/static/50v75duviqrhaj1h7a1v3gm6iv9d58en.zip',
	'https://nihcc.box.com/shared/static/oylbi4bmcnr2o65id2v9rfnqp16l3hp0.zip',
	'https://nihcc.box.com/shared/static/mw15sn09vriv3f1lrlnh3plz7pxt4hoo.zip',
	'https://nihcc.box.com/shared/static/zi68hd5o6dajgimnw5fiu7sh63kah5sd.zip',
	'https://nihcc.box.com/shared/static/3yiszde3vlklv4xoj1m7k0syqo3yy5ec.zip',
	'https://nihcc.box.com/shared/static/w2v86eshepbix9u3813m70d8zqe735xq.zip',
	'https://nihcc.box.com/shared/static/0cf5w11yvecfq34sd09qol5atzk1a4ql.zip',
	'https://nihcc.box.com/shared/static/275en88yybbvzf7hhsbl6d7kghfxfshi.zip',

	'https://nihcc.box.com/shared/static/l52tpmmkgjlfa065ow8czhivhu5vx27n.zip',
	'https://nihcc.box.com/shared/static/p89awvi7nj0yov1l2o9hzi5l3q183lqe.zip',
	'https://nihcc.box.com/shared/static/or9m7tqbrayvtuppsm4epwsl9rog94o8.zip',
	'https://nihcc.box.com/shared/static/vuac680472w3r7i859b0ng7fcxf71wev.zip',
	'https://nihcc.box.com/shared/static/pllix2czjvoykgbd8syzq9gq5wkofps6.zip',
	'https://nihcc.box.com/shared/static/2dn2kipkkya5zuusll4jlyil3cqzboyk.zip',
	'https://nihcc.box.com/shared/static/peva7rpx9lww6zgpd0n8olpo3b2n05ft.zip',
	'https://nihcc.box.com/shared/static/2fda8akx3r3mhkts4v6mg3si7dipr7rg.zip',
	'https://nihcc.box.com/shared/static/ijd3kwljgpgynfwj0vhj5j5aurzjpwxp.zip',
	'https://nihcc.box.com/shared/static/nc6rwjixplkc5cx983mng9mwe99j8oa2.zip',

	'https://nihcc.box.com/shared/static/rhnfkwctdcb6y92gn7u98pept6qjfaud.zip',
	'https://nihcc.box.com/shared/static/7315e79xqm72osa4869oqkb2o0wayz6k.zip',
	'https://nihcc.box.com/shared/static/4nbwf4j9ejhm2ozv8mz3x9jcji6knhhk.zip',
	'https://nihcc.box.com/shared/static/1lhhx2uc7w14bt70de0bzcja199k62vn.zip',
	'https://nihcc.box.com/shared/static/guho09wmfnlpmg64npz78m4jg5oxqnbo.zip',
	'https://nihcc.box.com/shared/static/epu016ga5dh01s9ynlbioyjbi2dua02x.zip',
	'https://nihcc.box.com/shared/static/b4ebv95vpr55jqghf6bthg92vktocdkg.zip',
	'https://nihcc.box.com/shared/static/byl9pk2y727wpvk0pju4ls4oomz9du6t.zip',
	'https://nihcc.box.com/shared/static/kisfbpualo24dhby243nuyfr8bszkqg1.zip',
	'https://nihcc.box.com/shared/static/rs1s5ouk4l3icu1n6vyf63r2uhmnv6wz.zip',

	'https://nihcc.box.com/shared/static/7tvrneuqt4eq4q1d7lj0fnafn15hu9oj.zip',
	'https://nihcc.box.com/shared/static/gjo530t0dgeci3hizcfdvubr2n3mzmtu.zip',
	'https://nihcc.box.com/shared/static/7x4pvrdu0lhazj83sdee7nr0zj0s1t0v.zip',
	'https://nihcc.box.com/shared/static/z7s2zzdtxe696rlo16cqf5pxahpl8dup.zip',
	'https://nihcc.box.com/shared/static/shr998yp51gf2y5jj7jqxz2ht8lcbril.zip',
	'https://nihcc.box.com/shared/static/kqg4peb9j53ljhrxe3l3zrj4ac6xogif.zip'
]

md5_link = 'https://nihcc.box.com/shared/static/q0f8gy79q2spw96hs6o4jjjfsrg17t55.txt'
urllib.request.urlretrieve(md5_link, OUTPUT_PATH + "MD5_checksums.txt")  # download the MD5 checksum file
for idx, link in enumerate(links):
	fn = OUTPUT_PATH+'Images_png_%02d.zip' % (idx+1)
	print('Downloading', fn, '...')
	urllib.request.urlretrieve(link, fn)  # download the zip file
	while not verify_md5('Images_png_%02d.zip' % (idx+1)):
		os.remove(fn)
		print('Re-Downloading', fn, '...')
		urllib.request.urlretrieve(link, fn)  # download the zip file
		
	with zipfile.ZipFile(fn, 'r') as zip_ref:
		print('Extracting', fn, '...')
		zip_ref.extractall(OUTPUT_PATH)

	os.remove(fn)
	print('Re-saving images correctly...')
	fix_save_images(OUTPUT_PATH + 'Images_png\\')

print("Done.")
