import requests
import json
from collections import Counter
import time
import re

def llamar_lmstudio_api(prompt, modelo, temperatura=0.7, timeout=60):
    """Llama a la API REST de LM Studio para generar una respuesta."""
    url = "http://localhost:1234/v1/completions"
    
    payload = {
        "model": modelo,
        "prompt": prompt,
        "temperature": temperatura,
        "max_tokens": 1024,
        "stop": None,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Enviando solicitud a la API de LM Studio (modelo: {modelo}, timeout: {timeout}s)...")
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            # Guardar respuesta completa para depuración
            with open("debug_respuesta_api.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            
            respuesta = result.get("choices", [{}])[0].get("text", "")
            return respuesta, None
        else:
            error_msg = f"Error en la API: {response.status_code} - {response.text}"
            print(error_msg)
            return "", error_msg
    
    except requests.exceptions.Timeout:
        return "", f"Timeout después de {timeout} segundos"
    except requests.exceptions.ConnectionError:
        return "", "Error de conexión. Verifica que LM Studio esté en ejecución en localhost:1234"
    except Exception as e:
        return "", f"Error inesperado: {str(e)}"

def verificar_modelos_disponibles():
    """Verifica qué modelos están disponibles en LM Studio."""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get("data", [])
            return [model["id"] for model in models]
        else:
            print(f"Error al obtener modelos: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error al verificar modelos disponibles: {str(e)}")
        return []

def ejecutar_lmstudio(prompt, modelo, num_muestras, temperatura=0.7):
    """Ejecuta LM Studio varias veces y recoge las respuestas."""
    respuestas = []
    respuestas_completas = []
    
    for i in range(num_muestras):
        print(f"\nEjecutando muestra {i+1}/{num_muestras}...")
        
        salida, error = llamar_lmstudio_api(prompt, modelo, temperatura, timeout=120)
        
        if error:
            print(f"Error en la ejecución {i+1}: {error}")
            continue
        
        print(f"Respuesta recibida. Longitud: {len(salida)} caracteres")
        respuestas_completas.append(salida)
        
        # Guardamos la respuesta completa para debugging
        with open(f"respuesta_completa_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(salida)
            
        # Intentar extraer la respuesta numérica para el problema de apretones de manos
        # Patrón de búsqueda: "apretón/apretones de manos", "en total", o números cerca de la palabra "total"
        patrones = [
            r"(?:hay|son|serían|existen|total de)[\s\w]*?(\d+)[\s\w]*?(?:apretones|apretón)",
            r"(?:apretones|apretón)[\s\w]*?(?:hay|son|serían|existen|total de)[\s\w]*?(\d+)",
            r"(?:respuesta|total)[\s\w]*?(?:es|son|serían|hay)[\s\w]*?(\d+)",
            r"(\d+)[\s\w]*?(?:apretones|apretón)[\s\w]*?(?:en total|total)",
            r"Respuesta:[\s\w]*?(\d+)"
        ]
        
        respuesta_encontrada = False
        for patron in patrones:
            match = re.search(patron, salida, re.IGNORECASE)
            if match:
                respuesta_num = match.group(1)
                respuestas.append(respuesta_num)
                print(f"Respuesta extraída: {respuesta_num}")
                respuesta_encontrada = True
                break
        
        if not respuesta_encontrada:
            # Buscar cualquier número en la parte final de la respuesta
            numeros = re.findall(r"(\d+)", salida[-200:])
            if numeros:
                respuesta_num = numeros[-1]  # Tomar el último número encontrado
                respuestas.append(respuesta_num)
                print(f"Respuesta alternativa (último número): {respuesta_num}")
            else:
                print("No se pudo extraer una respuesta numérica")
        
        # Pausa entre ejecuciones
        if i < num_muestras - 1:
            time.sleep(2)
    
    # Guardar todas las respuestas completas para análisis
    with open("todas_las_respuestas.txt", "w", encoding="utf-8") as f:
        for i, resp in enumerate(respuestas_completas):
            f.write(f"\n\n--- RESPUESTA {i+1} ---\n\n")
            f.write(resp)
    
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
    print("Verificando que el servidor de LM Studio esté en ejecución...")
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print(f"LM Studio está en ejecución")
        else:
            print("¡Advertencia! LM Studio respondió con un código de estado no esperado.")
    except Exception as e:
        print(f"¡Error! No se pudo conectar con LM Studio: {str(e)}")
        print("Asegúrate de que LM Studio esté en ejecución antes de continuar.")
        print("Debes iniciar LM Studio y activar el servidor local en la pestaña 'Server'.")
        exit(1)

    # Verificar modelos disponibles
    print("\nVerificando modelos disponibles...")
    modelos = verificar_modelos_disponibles()
    if modelos:
        print("Modelos disponibles:")
        for modelo in modelos:
            print(f"- {modelo}")
    else:
        print("No se pudieron obtener los modelos o no hay modelos cargados.")
        print("Debes cargar un modelo en LM Studio antes de continuar.")
        exit(1)

    # Prompt más explícito para el problema de apretones de manos
    prompt_largo = """Resuelve el siguiente problema paso a paso, mostrando tu razonamiento completo.

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

    # Usar un sistema de instrucciones más explícito para LM Studio si el endpoint lo permite
    try:
        # Verificar si el endpoint de chat existe y lo soporta
        chat_endpoint = "http://localhost:1234/v1/chat/completions"
        response = requests.post(chat_endpoint, json={"model": modelos[0], "messages": [{"role": "user", "content": "hola"}]}, timeout=5)
        if response.status_code == 200:
            print("\nDetectado soporte para el endpoint de chat. Usando formato de chat.")
            usar_chat = True
        else:
            print("\nNo se detectó soporte para el endpoint de chat. Usando formato de completions.")
            usar_chat = False
    except:
        print("\nNo se pudo detectar soporte para chat. Usando formato de completions.")
        usar_chat = False

    # Seleccionar el modelo
    if len(modelos) == 0:
        print("No se encontraron modelos. El script no puede continuar.")
        exit(1)
    
    print("\nModelos disponibles:")
    for i, modelo in enumerate(modelos):
        print(f"{i+1}. {modelo}")
    
    try:
        seleccion = int(input(f"\nSelecciona un modelo (1-{len(modelos)}): ")) - 1
        if 0 <= seleccion < len(modelos):
            modelo_seleccionado = modelos[seleccion]
        else:
            print("Selección inválida, usando el primer modelo disponible.")
            modelo_seleccionado = modelos[0]
    except ValueError:
        print("Entrada inválida, usando el primer modelo disponible.")
        modelo_seleccionado = modelos[0]
    
    print(f"\nUsando modelo: {modelo_seleccionado}")
    
    # Configuración para las ejecuciones
    num_muestras = int(input("\n¿Cuántas muestras deseas generar? (recomendado: 3-5): "))
    temperatura = float(input("\nIntroduce la temperatura (recomendado: 0.7-0.9): "))
    
    # Ejecutar LM Studio varias veces
    print(f"\nGenerando {num_muestras} respuestas con temperatura {temperatura}...")
    
    # Si hay soporte para chat, usamos ese formato
    if usar_chat:
        def llamar_lmstudio_chat_api(prompt, modelo, temperatura=0.7, timeout=60):
            url = "http://localhost:1234/v1/chat/completions"
            
            payload = {
                "model": modelo,
                "messages": [
                    {"role": "system", "content": "Eres un asistente matemático muy preciso. Resuelves problemas matemáticos paso a paso y siempre das la respuesta correcta."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperatura,
                "max_tokens": 1024,
                "stream": False
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                
                if response.status_code == 200:
                    result = response.json()
                    respuesta = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return respuesta, None
                else:
                    return "", f"Error en la API: {response.status_code} - {response.text}"
            except Exception as e:
                return "", f"Error: {str(e)}"
        
        # Redefinir la función llamar_lmstudio_api para usar el formato de chat
        llamar_lmstudio_api = llamar_lmstudio_chat_api
    
    # Ejecutar LM Studio varias veces
    respuestas = ejecutar_lmstudio(prompt_largo, modelo_seleccionado, num_muestras, temperatura)
    
    # Mostrar todas las respuestas
    print("\nRespuestas numéricas obtenidas:")
    for i, resp in enumerate(respuestas):
        print(f"\n--- Respuesta {i+1} ---")
        print(resp)
    
    # Mostrar la respuesta más consistente
    respuesta_consistente = obtener_respuesta_consistente(respuestas)
    
    print("\nRespuesta más consistente:")
    print(f"Hay {respuesta_consistente} apretones de manos en total.")
    
    # Validación de la respuesta correcta
    # Para 5 personas, la respuesta correcta es 10 apretones de manos: (5 * 4) / 2 = 10
    print("\nVerificación de la respuesta:")
    n = 5  # número de personas
    respuesta_correcta = (n * (n - 1)) // 2
    print(f"La respuesta correcta para {n} personas es: {respuesta_correcta} apretones de manos")
    
    if respuesta_consistente == str(respuesta_correcta):
        print("✅ La respuesta consistente coincide con la respuesta correcta.")
    else:
        print(f"❌ La respuesta consistente ({respuesta_consistente}) no coincide con la respuesta correcta ({respuesta_correcta}).")