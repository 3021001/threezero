#Prepare Testing Data
test_filenames = os.listdir("./data/test1/")
test_df = pd.DataFrame({
  'filename':test_filenames
})
nb_samples = test_df.shape[0]

//평가 데이터 준비
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
  test_df,
  ".data/test1/",
  x_col='filename',
  y_col=None,
  class_mode=None,
  target_size=IMAGE_SIZE,
  batch_size=batch_size,
  shuffle=False
)

//모델 예측
#Predict
predict = mode.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

//평가 생성
test_df['category'] = np.argmax(predict, axis=-1)

//레이블 변환, dog=1, cat=0으로 변경
test_df['category'] = test_df'['category'].replace({'dog':1,'cat':0})

//정답비율 확인하기
#Virtualize Result
test_df['category'].value_counts().plot.bar()

//정답 확인
#See predicated result with images
sample_test = test_df.head(18)
sample_test.head()
plt. figure(figsize(12, 24))
for index, row in sample_test.iterrows():
  filename = row['filename']
  category = row['category']
  img = load_img("./data/test1/"+filename, target_size=IMAGE_SIZE)
  plt.subplot(6, 3, index+1)
  plt.imshow(img)
  plt.xlabel(filename + '(' + "{}".format(category) + ')' )
  plt.tight_layout()
  plt.show()
