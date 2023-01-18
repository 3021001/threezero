//학습 시작
#Fit model
epochs=3 if FAST_RUN else 50
history = model.fit_generator(
  train_generator,
  epochs=epochs,
  validation_data=validation_generator,
  validation_steps=total_validate//batch_size,
  steps_per_epoch=total_train//batch_size,
  callbacks=callbacks
)

//모델 저장
#Save Model
model.save_weights("model.h5")

//학습 내용 확인하기
#Virtualize Training
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="Validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r', label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
