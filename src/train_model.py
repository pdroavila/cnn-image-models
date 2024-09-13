import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import numpy as np

def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_dir, val_dir, batch_size=32, epochs=10):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size)

    return history

if __name__ == "__main__":
    # Obter o diretório do script atual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Definir caminhos corretos
    project_dir = os.path.dirname(script_dir)  # Diretório pai do script
    train_dir = os.path.join(project_dir, 'data', 'train')
    val_dir = os.path.join(project_dir, 'data', 'val')
    models_dir = os.path.join(project_dir, 'models')

    # Criar o diretório models se não existir
    os.makedirs(models_dir, exist_ok=True)

    # Imprimir caminhos para verificação
    print(f"Diretório do script: {script_dir}")
    print(f"Diretório do projeto: {project_dir}")
    print(f"Diretório de treinamento: {train_dir}")
    print(f"Diretório de validação: {val_dir}")
    print(f"Diretório de modelos: {models_dir}")

    # Criar e treinar o modelo
    model = create_model()
    history = train_model(model, train_dir, val_dir, batch_size=32, epochs=10)
    
    # Salvar o modelo
    model_path = os.path.join(models_dir, 'modelo_rg.keras')
    try:
        model.save(model_path)
        print(f"Modelo salvo com sucesso em: {model_path}")
        
        if os.path.exists(model_path):
            print(f"Tamanho do arquivo do modelo: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        else:
            print("Erro: O arquivo do modelo não foi criado.")
    except Exception as e:
        print(f"Erro ao salvar o modelo: {str(e)}")
    
    # Verificar o conteúdo do diretório de modelos
    print("\nConteúdo do diretório 'models':")
    for file in os.listdir(models_dir):
        print(file)
