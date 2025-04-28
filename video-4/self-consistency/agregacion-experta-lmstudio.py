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
        "max_tokens": 2048,  # Aumentado para respuestas más largas en evaluación
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

# La corrección se enfoca en mejorar el prompt y la captura de respuestas

# Modificación 1: Mejorar el prompt para asegurarnos que el modelo responda al problema específico
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
        
        # Extraer la respuesta - Mejoramos la extracción
        respuesta_match = re.search(r"Respuesta:\s*(.*?)(?:\n\n|$)", salida, re.DOTALL)
        if respuesta_match:
            respuesta = respuesta_match.group(1).strip()
            # Verificar que la respuesta contiene un número
            if re.search(r'\d+', respuesta):
                respuestas.append(respuesta)
                print(f"Respuesta extraída: {respuesta[:50]}..." if len(respuesta) > 50 else respuesta)
            else:
                print("La respuesta no contiene números, puede ser incorrecta")
                # Aún así la agregamos para análisis
                respuestas.append(respuesta)
        else:
            # Búsqueda alternativa para respuestas sin etiqueta formal
            numeros_en_texto = re.findall(r'\b(\d+)\s+apretones de manos', salida)
            if numeros_en_texto:
                respuesta_alt = f"{numeros_en_texto[0]} apretones de manos"
                respuestas.append(respuesta_alt)
                print(f"Respuesta alternativa extraída: {respuesta_alt}")
            else:
                ultimas_lineas = salida.strip().split('\n')[-3:]
                respuesta_alt = ' '.join(ultimas_lineas)
                respuestas.append(respuesta_alt)
                print(f"Usando últimas líneas: {respuesta_alt[:50]}..." if len(respuesta_alt) > 50 else respuesta_alt)
        
        # Guardamos la respuesta completa para debugging
        with open(f"respuesta_completa_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(salida)
        
        # Pausa entre ejecuciones
        if i < num_muestras - 1:
            time.sleep(2)
    
    return respuestas, respuestas_completas

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

def crear_meta_prompt(problema, respuestas_completas):
    """Crea un prompt más conciso para que el modelo evalúe múltiples soluciones."""
    # Truncamos las respuestas si son muy largas para evitar el error de contexto
    respuestas_truncadas = []
    for resp in respuestas_completas:
        # Limitamos cada respuesta a aproximadamente 600 tokens (unas 800 palabras)
        palabras = resp.split()
        if len(palabras) > 800:
            resp_truncada = ' '.join(palabras[:800]) + "... [respuesta truncada]"
            respuestas_truncadas.append(resp_truncada)
        else:
            respuestas_truncadas.append(resp)
    
    meta_prompt = f"""INSTRUCCIÓN: RESPONDE EN ESPAÑOL.

Como experto matemático, analiza estas soluciones al problema y determina cuál es correcta.

PROBLEMA:
{problema}

A continuación se presentan {len(respuestas_truncadas)} soluciones. Identifica la correcta.
"""

    for i, respuesta in enumerate(respuestas_truncadas):
        meta_prompt += f"\n\nSOLUCIÓN {i+1}:\n"
        meta_prompt += respuesta

    meta_prompt += """

Evalúa las soluciones y responde:
1. ¿Cuál es la respuesta correcta al problema?
2. Explica brevemente por qué.

Tu evaluación final:
"""
    
    return meta_prompt

def intentar_chat_api(prompt, modelo, temperatura=0.7, timeout=60):
    """Intenta usar la API de chat si está disponible."""
    url = "http://localhost:1234/v1/chat/completions"
    
    payload = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": "Eres un experto matemático y tu respuesta DEBE estar en español. Analiza con cuidado y responde con precisión."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperatura,
        "max_tokens": 2048,
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
            return respuesta, None, True
        else:
            # Si falla, indicamos que la API de chat no está disponible
            return "", "", False
    except:
        return "", "", False

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
        print("No se pudieron obtener los modelos o no hay modelos instalados.")
        print("Debes cargar un modelo en LM Studio antes de continuar.")
        exit(1)

    # Mostrar modelos disponibles para selección
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

    problema = "Hay 5 personas en una habitación. Cada persona saluda a todas las demás con un apretón de manos. ¿Cuántos apretones de manos hay en total?"
    
    prompt_mejorado = """Resuelve el siguiente problema matemático paso a paso, mostrando tu razonamiento claro.

Problema: Hay 5 personas en una habitación. Cada persona saluda a todas las demás con un apretón de manos. ¿Cuántos apretones de manos hay en total?

Piensa de forma ordenada. Para cada persona, cuenta a cuántas otras personas debe saludar, teniendo en cuenta que cada apretón de manos ocurre exactamente una vez entre dos personas.

Razonamiento:
"""
    
    # Configuración para las ejecuciones
    num_muestras = int(input("\n¿Cuántas muestras deseas generar? (recomendado: 5-10): "))
    temperatura = float(input("\nIntroduce la temperatura para las muestras (recomendado: 0.7-0.9): "))
    
    # Ejecutar LM Studio varias veces con el prompt mejorado
    print(f"\nGenerando {num_muestras} respuestas con temperatura {temperatura}...")
    respuestas, respuestas_completas = ejecutar_lmstudio(prompt_mejorado, modelo_seleccionado, num_muestras, temperatura)
    
    # Mostrar todas las respuestas
    print("\nRespuestas obtenidas:")
    for i, resp in enumerate(respuestas):
        print(f"\n--- Respuesta {i+1} ---")
        print(resp[:200] + "..." if len(resp) > 200 else resp)
    
    # Mostrar la respuesta más consistente (método estadístico)
    respuesta_consistente = obtener_respuesta_consistente(respuestas)
    
    print("\nRespuesta más consistente (por frecuencia):")
    print(respuesta_consistente)
    
    # Crear un meta-prompt para evaluación
    print("\nCreando meta-prompt para evaluación experta...")
    meta_prompt = crear_meta_prompt(problema, respuestas_completas)
    
    # Guardar el meta-prompt para referencia
    with open("meta_prompt.txt", "w", encoding="utf-8") as f:
        f.write(meta_prompt)
    
    print("Meta-prompt creado y guardado en 'meta_prompt.txt'")
    
    # Intentar usar la API de chat primero
    print("\nVerificando si la API de chat está disponible...")
    meta_respuesta, meta_error, chat_disponible = intentar_chat_api(meta_prompt, modelo_seleccionado, temperatura=0.2)
    
    if chat_disponible:
        print("La API de chat está disponible. Usando este formato para mejor evaluación...")
    else:
        print("La API de chat no está disponible. Usando la API estándar...")
        # Ejecutar el meta-análisis con temperatura más baja
        print("Solicitando evaluación experta al modelo...")
        meta_respuesta, meta_error = llamar_lmstudio_api(meta_prompt, modelo_seleccionado, temperatura=0.2, timeout=180)
    
    if meta_error:
        print(f"Error al realizar la evaluación experta: {meta_error}")
    else:
        print("\n======= EVALUACIÓN EXPERTA DEL MODELO =======")
        print(meta_respuesta)
        print("=============================================")
        
        # Verificar si la respuesta está en español
        if any(palabra in meta_respuesta.lower() for palabra in ["the", "is", "correct", "solution", "answer"]) and not any(palabra in meta_respuesta.lower() for palabra in ["la", "es", "correcta", "solución", "respuesta"]):
            print("\n¡ADVERTENCIA! La respuesta parece estar en inglés a pesar de la instrucción.")
            print("Intentando obtener respuesta en español con instrucciones más claras...")
            
            # Crear un nuevo prompt aún más explícito
            nuevo_meta_prompt = f"""IMPORTANTE: DEBES RESPONDER ÚNICAMENTE EN ESPAÑOL. NO EN INGLÉS.

{meta_prompt}

NOTA FINAL: TU RESPUESTA DEBE ESTAR 100% EN ESPAÑOL. NO USES INGLÉS EN ABSOLUTO.
"""
            if chat_disponible:
                meta_respuesta, meta_error, _ = intentar_chat_api(nuevo_meta_prompt, modelo_seleccionado, temperatura=0.1, timeout=180)
            else:
                meta_respuesta, meta_error = llamar_lmstudio_api(nuevo_meta_prompt, modelo_seleccionado, temperatura=0.1, timeout=180)
            
            if meta_error:
                print(f"Error al realizar el segundo intento: {meta_error}")
            else:
                print("\n======= EVALUACIÓN EXPERTA (SEGUNDO INTENTO) =======")
                print(meta_respuesta)
                print("=============================================")
        
        # Guardar la meta-respuesta
        with open("meta_respuesta.txt", "w", encoding="utf-8") as f:
            f.write(meta_respuesta)
            
    # Comparación final
    print("\n=== COMPARACIÓN DE TÉCNICAS ===")
    print(f"1. Respuesta más consistente (estadística): {respuesta_consistente}")
    print("2. Evaluación experta (meta-prompt): Ver el análisis detallado arriba")
    
    # Verificación matemática
    n = 5  # Número de personas
    respuesta_correcta = (n * (n - 1)) // 2  # Fórmula matemática
    print(f"\nLa respuesta matemáticamente correcta es: {respuesta_correcta} apretones de manos")