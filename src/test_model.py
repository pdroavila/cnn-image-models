import os
import tensorflow as tf
import numpy as np


def validate_rg(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return prediction[0][0] > 0.5

if __name__ == "__main__":

    # Definir caminhos corretos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # Diretório pai do script
    train_dir = os.path.join(project_dir, 'data', 'train')
    val_dir = os.path.join(project_dir, 'data', 'val')
    models_dir = os.path.join(project_dir, 'models')
    model_path = os.path.join(models_dir, 'modelo_final.h5')
    test = os.path.join(project_dir, 'imagens', 'rg_pedro.jpg')

    # Carregar e usar o modelo
    try:
        modelo_carregado = tf.keras.models.load_model(model_path)
        print("Modelo carregado com sucesso.")
        
        # Exemplo de validação
        exemplo_path = test
        if os.path.exists(exemplo_path):
            is_rg = validate_rg(exemplo_path, modelo_carregado)
            print("A imagem é um RG:", is_rg)
        else:
            print(f"Aviso: Imagem de exemplo não encontrada em {exemplo_path}")
    except Exception as e:
        print(f"Erro ao carregar ou usar o modelo: {e}")