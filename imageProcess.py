#-*-coding:utf-8-*-
import matplotlib.pyplot as plt 
import tensorflow as tf 
#线程
from threading import Thread
#显示中文
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

#读取图像原始数据
image_raw_data = tf.gfile.FastGFile("./picture/cat.jpg", 'r').read()
print(type(image_raw_data))

with tf.Session() as sess:
	#图片数据解码
	img_data = tf.image.decode_jpeg(image_raw_data, channels=3)
	print(type(img_data))
	print(img_data.shape)
	# print(img_data.eval())
	#根据解码数据，输出图片
	plt.imshow(img_data.eval())
	# plt.show()
	#图像重新编码
	encoded_image = tf.image.encode_jpeg(img_data)
	print(type(encoded_image))
	print(encoded_image)
	with tf.gfile.GFile("./picture/output", "wb") as f:
		f.write(encoded_image.eval())

	#图片类型转化：实数转为浮点数
	img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
	print(img_data.eval())
	#保留图片内容，改变图片尺寸
	##双线性插值法调整图片大小
	resized_method0 = tf.image.resize_images(img_data, [300, 300], method=0)
	##最邻近法调整图片大小
	resized_method1 = tf.image.resize_images(img_data, [300, 300], method=1)
	##双三次插值法调整图片大小
	resized_method2 = tf.image.resize_images(img_data, [300, 300], method=2)
	##面积插值法调整图片大小
	resized_method3 = tf.image.resize_images(img_data, [300, 300], method=3)
	#调整图片集合
	resizes = []
	method_desc = ['Bilinear', 'Nearest neighbor', 'Bicubic', 'Area']
	print(type(method_desc[0]))
	for i in range(4):
		resized_method = tf.image.resize_images(img_data, [300, 300], method=i)
		resizes.append(resized_method)
	print(len(resizes))
	for i in range(len(resizes)):
		plt.gcf().canvas.set_window_title('不同方法处理图片')
		plt.gcf().suptitle("Interpolation")
		# print(resizes[i].eval())
		print(i)
		# plt.figure(i)
		# plt.subplot(2, 2, i+1).set_title("method=%d"%i)
		plt.subplot(2, 2, i+1).set_title(method_desc[i])

		#显示
		plt.imshow(resizes[i].eval())

	#保存图片
	plt.savefig('./picture/resized.jpg')
	#屏幕显示
	plt.ion()
	plt.show()
	#显示时间:秒数
	plt.pause(5)
	#关闭显示
	plt.close()
	
	# print(resized_method0.eval())
	# plt.imshow(resized_method0.eval())
	# plt.show()
	#裁剪图片：截取指定尺寸
	croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
	print(type(croped))
	padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
	print(type(padded))
	#绘制画图区、加标题、显示、保存
	plt.figure(1)
	plt.title('Crop')
	plt.imshow(croped.eval())
	plt.savefig('./picture/crop.jpg')

	plt.figure(2)
	plt.title('Padding')
	plt.imshow(padded.eval())
	plt.savefig('./picture/pad.jpg')

	plt.ion()
	plt.show()
	plt.pause(10)
	plt.close()
	#调整图片大小：比例
	#0.5调整比例[0,1]
	central_cropped = tf.image.central_crop(img_data, 0.5)

	plt.figure(1)
	plt.title('Crop para')
	plt.imshow(central_cropped.eval())
	plt.savefig('./picture/cropparam.jpg')
	plt.ion()
	plt.show()
	plt.pause(10)
	plt.close()
	#旋转图片
	##上下旋转：180度
	flip0 = tf.image.flip_up_down(img_data)
	##左右旋转
	flip1 = tf.image.flip_left_right(img_data)
	##对角线旋转
	flip2 = tf.image.transpose_image(img_data)

	plt.subplot(1, 3, 1).set_title('up_down')
	plt.imshow(flip0.eval())

	plt.subplot(1, 3, 2).set_title('left_right')
	plt.imshow(flip1.eval())

	plt.subplot(1, 3, 3).set_title('transpose')
	plt.imshow(flip2.eval())

	plt.savefig('./picture/flip.jpg')
	plt.ion()
	plt.show()
	plt.pause(10)
	plt.close()

	#50%概率旋转图片
	flip1 = tf.image.random_flip_up_down(img_data)
	flip2 = tf.image.random_flip_left_right(img_data)

	#调整图片色彩
	##调整亮度:变黑
	brightness0 = tf.image.adjust_brightness(img_data, -0.5)
	##图片亮度拉回到[0,1]
	brightness1 = tf.clip_by_value(brightness0, 0.0, 1.0)


	##图片亮度：+0.6
	brightness2 = tf.image.adjust_brightness(img_data, 0.5)
	##随机增加亮度
	# brightness3 = tf.image.random_brightness(img_data, max_delta)
	##调整亮度：倍数
	##亮度x0.8
	brightness4 = tf.image.adjust_contrast(img_data, 0.8)
	##亮度x2
	brightness5 = tf.image.adjust_contrast(img_data, 2)
	##随机调整亮度
	# brightness6 = tf.image.random_contrast(img_data, lower, upper)


	##调整色相
	hue0 = tf.image.adjust_hue(img_data, 0.2)
	##随机调整色相
	# hue1 = tf.image.random_hue(img_data, max_delta)


	##调整饱和度
	##调整到-2
	saturation0 = tf.image.adjust_saturation(img_data, -2)
	##调整到+2
	saturation1 = tf.image.adjust_saturation(img_data, 2)
	##随机调整饱和度
	# saturation2 = tf.image.random_saturation(img_data, lower, upper)

	##图像均值为0，方差为1
	changed = tf.image.per_image_standardization(img_data)

	subplot(3, 3, 1).set_title('1')
	plt.imshow(brightness0.eval())

	subplot(3, 3, 2).set_title('2')
	plt.imshow(brightness1.eval())

	subplot(3, 3, 3).set_title('3')
	plt.imshow(brightness2.eval())

	subplot(3, 3, 4).set_title('4')
	plt.imshow(brightness4.eval())

	subplot(3, 3, 5).set_title('5')
	plt.imshow(brightness5.eval())

	subplot(3, 3, 6).set_title('6')
	plt.imshow(hue0.eval())

	subplot(3, 3, 7).set_title('7')
	plt.imshow(saturation0.eval())

	subplot(3, 3, 8).set_title('8')
	plt.imshow(saturation1.eval())

	subplot(3, 3, 9).set_title('9')
	plt.imshow(changed.eval())

	plt.savefig('./picture/mix.jpg')

	plt.ion()
	plt.show()
	plt.pause(10)
	plt.close()
	#标注框
	##img_data已转化为float32
	resized_method0 = tf.image.resize_images(img_data, [180, 267], method=0)
	# plt.imshow(resized_method0.eval())
	# plt.ion()
	# plt.show()
	# plt.pause(10)
	# plt.close()
	##img_data是三维矩阵，改变成四维矩阵
	batched = tf.expand_dims(resized_method0, 0)
	##标注框[y_min, x_min,y_max, x_max]
	boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7],[0.35, 0.47, 0.5, 0.56]]])
	##添加框
	bounding_box = tf.image.draw_bounding_boxes(batched, boxes)
	print(type(bounding_box))
	print(bounding_box.eval())
	print(type(bounding_box[0].eval()))
	print(bounding_box[0].eval())
	# plt.imshow(bounding_box.eval())
	plt.imshow(bounding_box[0].eval())

	plt.ion()
	plt.show()
	plt.pause(10)
	plt.close()




























