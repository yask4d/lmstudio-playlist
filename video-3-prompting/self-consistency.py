import requests
import json
from collections import Counter
import time
import re

def llamar_ollama_api(prompt, modelo, temperatura=0.7, timeout=60):
    """Llama a la API REST de Ollama para generar una respuesta."""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": modelo,
        "prompt": prompt,
        "temperature": temperatura,
        "stream": False  # No queremos streaming para simplificar
    }
    
    try:
        print(f"Enviando solicitud a la API de Ollama (modelo: {modelo}, timeout: {timeout}s)...")
        response = requests.post(url, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", ""), None
        else:
            error_msg = f"Error en la API: {response.status_code} - {response.text}"
            print(error_msg)
            return "", error_msg
    
    except requests.exceptions.Timeout:
        return "", f"Timeout después de {timeout} segundos"
    except requests.exceptions.ConnectionError:
        return "", "Error de conexión. Verifica que Ollama esté en ejecución en localhost:11434"
    except Exception as e:
        return "", f"Error inesperado: {str(e)}"

def verificar_modelos_disponibles():
    """Verifica qué modelos están disponibles en Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        else:
            print(f"Error al obtener modelos: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error al verificar modelos disponibles: {str(e)}")
        return []

def ejecutar_ollama(prompt, modelo, num_muestras, temperatura=0.7):
    """Ejecuta Ollama varias veces y recoge las respuestas."""
    respuestas = []
    
    for i in range(num_muestras):
        print(f"\nEjecutando muestra {i+1}/{num_muestras}...")
        
        salida, error = llamar_ollama_api(prompt, modelo, temperatura, timeout=120)
        
        if error:
            print(f"Error en la ejecución {i+1}: {error}")
            continue
        
        print(f"Respuesta recibida. Longitud: {len(salida)} caracteres")
        
        # Extraer la respuesta
        ultimo_razonamiento = salida.rfind("Razonamiento:")
        if ultimo_razonamiento >= 0:
            texto_respuesta = salida[ultimo_razonamiento:]
            
            # Buscar "Respuesta:" después del "Razonamiento:"
            match_respuesta = re.search(r"Respuesta:(.*?)(?:\n\n|$)", texto_respuesta, re.DOTALL)
            if match_respuesta:
                respuesta = match_respuesta.group(1).strip()
                respuestas.append(respuesta)
                print(f"Respuesta extraída: {respuesta[:50]}..." if len(respuesta) > 50 else respuesta)
            else:
                # Si no encontramos "Respuesta:", guardamos las últimas líneas
                ultimas_lineas = texto_respuesta.strip().split('\n')[-3:]
                respuesta_alt = ' '.join(ultimas_lineas)
                respuestas.append(respuesta_alt)
                print(f"Respuesta alternativa: {respuesta_alt[:50]}..." if len(respuesta_alt) > 50 else respuesta_alt)
        else:
            # Si no encontramos ni siquiera "Razonamiento:", tomamos las últimas líneas
            ultimas_lineas = salida.strip().split('\n')[-5:]
            respuesta_alt = ' '.join(ultimas_lineas)
            respuestas.append(respuesta_alt)
            print(f"Usando últimas líneas como respuesta: {respuesta_alt[:50]}..." if len(respuesta_alt) > 50 else respuesta_alt)
        
        # Guardamos la respuesta completa para debugging
        with open(f"respuesta_completa_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(salida)
        
        # Pausa entre ejecuciones
        if i < num_muestras - 1:
            time.sleep(2)
    
    return respuestas

def obtener_respuesta_consistente(respuestas):
    """Determina la respuesta más frecuente en una lista de respuestas."""
    if not respuestas:
        return "No se pudo obtener ninguna respuesta."

    contador = Counter(respuestas)
    respuesta_mas_comun = contador.most_common(1)[0][0]
    frecuencia = contador.most_common(1)[0][1]
    total = len(respuestas)
    
    print(f"\nEstadísticas de consistencia:")
    print(f"- Respuesta más común apareció {frecuencia} de {total} veces ({frecuencia/total*100:.1f}%)")
    
    return respuesta_mas_comun

if __name__ == '__main__':
    print("Verificando que el servidor de Ollama esté en ejecución...")
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version = response.json().get("version", "desconocida")
            print(f"Ollama está en ejecución (versión: {version})")
        else:
            print("¡Advertencia! Ollama respondió con un código de estado no esperado.")
    except Exception as e:
        print(f"¡Error! No se pudo conectar con Ollama: {str(e)}")
        print("Asegúrate de que Ollama esté en ejecución antes de continuar.")
        print("Puedes iniciar Ollama ejecutando simplemente 'ollama serve' en otra terminal.")
        exit(1)

    # Verificar modelos disponibles
    print("\nVerificando modelos disponibles...")
    modelos = verificar_modelos_disponibles()
    if modelos:
        print("Modelos disponibles:")
        for modelo in modelos:
            print(f"- {modelo}")
    else:
        print("No se pudieron obtener los modelos o no hay modelos instalados.")
        print("Puedes instalar modelos con 'ollama pull <nombre-modelo>'")
        exit(1)

    # Prompt para el problema de apretones de manos
    prompt_largo = """Resuelve el siguiente problema paso a paso, mostrando tu razonamiento.

Problema: María tiene 5 manzanas y Juan le da otras 3. ¿Cuántas manzanas tiene María en total?
Razonamiento:
1. María comienza con 5 manzanas.
2. Juan le da 3 manzanas adicionales.
3. Para encontrar el total, sumamos las manzanas iniciales de María más las que recibió de Juan: 5 + 3 = 8
Respuesta: María tiene 8 manzanas en total.

Problema: Un tren sale de la estación A a una velocidad de 60 km/h y otro tren sale de la estación B (a 300 km de A) a una velocidad de 80 km/h. ¿Cuánto tardarán en encontrarse?
Razonamiento:
1. La distancia total entre las estaciones es de 300 km.
2. Los trenes se acercan el uno al otro a una velocidad combinada de 60 + 80 = 140 km/h.
3. Para encontrar el tiempo que tardan en encontrarse, dividimos la distancia total por la velocidad combinada: 300 / 140 ≈ 2.14 horas.
Respuesta: Los trenes tardarán aproximadamente 2.14 horas en encontrarse.

Problema: Hay 5 personas en una habitación. Cada persona saluda a todas las demás con un apretón de manos. ¿Cuántos apretones de manos hay en total?
Razonamiento:
"""

    # Seleccionar el modelo
    modelo_seleccionado = None
    if modelos:
        # Intentar encontrar un modelo gemma
        for m in modelos:
            if "gemma3:27b" in m.lower():
                modelo_seleccionado = m
                break
        
        # Si no hay gemma, usar el primer modelo disponible
        if not modelo_seleccionado:
            modelo_seleccionado = modelos[0]
    else:
        print("No se encontraron modelos. El script no puede continuar.")
        exit(1)
    
    print(f"\nUsando modelo: {modelo_seleccionado}")
    
    # Configuración para las ejecuciones
    num_muestras = 3
    temperatura = 0.8  # Un poco más de temperatura para generar variedad
    
    # Ejecutar Ollama varias veces
    print(f"\nGenerando {num_muestras} respuestas con temperatura {temperatura}...")
    respuestas = ejecutar_ollama(prompt_largo, modelo_seleccionado, num_muestras, temperatura)
    
    # Mostrar todas las respuestas
    print("\nRespuestas obtenidas:")
    for i, resp in enumerate(respuestas):
        print(f"\n--- Respuesta {i+1} ---")
        print(resp[:200] + "..." if len(resp) > 200 else resp)
    
    # Mostrar la respuesta más consistente
    respuesta_consistente = obtener_respuesta_consistente(respuestas)
    
    print("\nRespuesta más consistente:")
    print(respuesta_consistente)