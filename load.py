from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

model = load_model(r'C:\env\models\fp32_model\model.keras')
model.summary()


