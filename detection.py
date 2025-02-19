from functions import train, prediction

model = train()

prediction("./image.jpg", model)