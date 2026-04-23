from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from io import BytesIO
import os

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        #print("POST:", request.POST)
        #print("FILES:", request.FILES)
     form = ImageUploadForm(request.POST, request.FILES)
        
     if form.is_valid():
            #print("FORM VALID")
         img_file = form.cleaned_data['image']
         img_file = BytesIO(img_file.read())
         img = load_img(img_file, target_size=(224, 224))
         img_array = img_to_array(img)
         img_array = img_array.reshape((1, 224, 224, 3))
         img_array = preprocess_input(img_array)
         model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
         model = load_model(model_path)
         result = model.predict(img_array)
         prediction = decode_predictions(result, top=5)
         top5 = prediction[0]
         img_data = request.POST.get('img_data') 
         return render(request, 'home.html', {'form': form, 'top5': top5, 'img_data': img_data})
     else:
          #print("FORM INVALID")
          #print("ERRORS:", form.errors)
          #print("FILES:", request.FILES)
          return render(request, 'home.html', {'form': form})