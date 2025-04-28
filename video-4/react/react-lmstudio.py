import requests
import json
import time
import re
import os
from typing import Dict, List, Tuple, Optional, Any

def llamar_lmstudio_api(prompt: str, temperatura: float = 0.7, timeout: int = 120) -> Tuple[str, Optional[str]]:
    """Llama a la API REST de LM Studio para generar una respuesta."""
    url = "http://localhost:1234/v1/completions"
    
    payload = {
        "prompt": prompt,
        "temperature": temperatura,
        "max_tokens": 1024,
        "stop": ["<|endoftext|>"],
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Enviando solicitud a la API de LM Studio (timeout: {timeout}s)...")
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            # La respuesta estará en el primer elemento de "choices" en el campo "text"
            return result.get("choices", [{}])[0].get("text", ""), None
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

def verificar_lmstudio_disponible() -> bool:
    """Verifica si LM Studio está en ejecución."""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def procesar_reacciones(respuesta: str) -> Dict[str, List[str]]:
    """Extrae las secuencias de Pensamiento, Acción y Observación del texto de respuesta ReAct."""
    resultado = {
        "pensamiento": [],
        "accion": [],
        "observacion": []
    }
    
    # Patrones para detectar secciones de ReAct
    patrones = {
        "pensamiento": r"Pensamiento:(.+?)(?=Acción:|Observación:|$)",
        "accion": r"Acción:(.+?)(?=Pensamiento:|Observación:|$)",
        "observacion": r"Observación:(.+?)(?=Pensamiento:|Acción:|$)"
    }
    
    # Extraer todas las instancias de cada patrón
    for tipo, patron in patrones.items():
        matches = re.findall(patron, respuesta, re.DOTALL)
        for match in matches:
            resultado[tipo].append(match.strip())
    
    return resultado

def simular_observacion(accion: str, problema: Dict[str, Any]) -> str:
    """Simula observaciones basadas en acciones específicas para el problema."""
    accion = accion.lower()
    
    # Simulación para problemas de búsqueda en Internet
    if "buscar" in accion:
        if "población" in accion and "parís" in accion:
            return "Según datos recientes, la población de París es de aproximadamente 2.16 millones en la ciudad propiamente dicha, y más de 12 millones en el área metropolitana."
        elif "capital" in accion and "francia" in accion:
            return "La capital de Francia es París."
        elif "monte" in accion and "everest" in accion:
            return "El Monte Everest es la montaña más alta del mundo con una altura de 8,848.86 metros sobre el nivel del mar."
        elif "equipo" in accion and "escalada" in accion:
            return "El equipo básico para escalar montañas incluye: piolets, crampones, cuerdas, mosquetones, arnés, casco, botas de montaña, ropa térmica, tienda de campaña especial para condiciones extremas, sacos de dormir para temperaturas bajo cero, y equipo de oxígeno para alturas superiores a 8000m."
        elif "temporada" in accion and ("everest" in accion or "escalada" in accion):
            return "Las mejores temporadas para escalar el Everest son finales de abril a principios de junio (pre-monzón) y septiembre a octubre (post-monzón). La mayoría de expediciones se realizan en mayo cuando las condiciones son más estables."
        else:
            return "La búsqueda no produjo resultados relevantes."
    
    # Simulación para problemas de cálculo
    elif any(palabra in accion for palabra in ["calcular", "computar", "resolver"]):
        if "área" in accion and "triángulo" in accion:
            if "base" in accion and "6" in accion and "altura" in accion and "8" in accion:
                return "El área del triángulo con base 6 cm y altura 8 cm es: (6 × 8) ÷ 2 = 24 cm²."
            else:
                return "El área de un triángulo se calcula con la fórmula: A = (base × altura) ÷ 2"
        elif "área" in accion and "círculo" in accion:
            if "radio" in accion and "5" in accion:
                return "El área del círculo con radio 5 cm es: π × 5² = π × 25 = 78.54 cm²."
            else:
                return "El área de un círculo se calcula con la fórmula: A = π × r²"
        elif "comparar" in accion and ("triángulo" in accion or "círculo" in accion):
            return "El área del triángulo es 24 cm² y el área del círculo es 78.54 cm². Por lo tanto, el círculo tiene un área aproximadamente 3.27 veces mayor que el triángulo."
        else:
            return "No tengo información suficiente para realizar este cálculo específico."
    
    # Simulación para otras acciones específicas del problema
    elif problema.get("tipo") == "planeacion_viaje":
        if "vuelos" in accion or "avión" in accion:
            return "Hay vuelos disponibles desde Madrid a París los lunes, miércoles y viernes, con precios desde 120€."
        elif "tren" in accion:
            return "El tren de alta velocidad conecta Madrid con París en 10 horas, con precios desde 180€."
        elif "alojamiento" in accion or "hotel" in accion:
            return "Hay varios hoteles disponibles en París, con precios entre 80€ y 300€ por noche dependiendo de la ubicación y categoría."
        elif "presupuesto" in accion or "1000€" in accion:
            return "Con 1000€ para 5 días, podrías gastar aproximadamente: 200€ en transporte ida y vuelta, 500€ en alojamiento (100€/noche), y 300€ para comidas y atracciones (60€/día)."
        else:
            return "No hay información específica disponible para esta consulta sobre el viaje."
    
    # Respuesta por defecto
    else:
        return "No se pudo procesar la acción solicitada. Por favor, especifica mejor lo que quieres hacer."

def ejecutar_react(prompt_base: str, problema: Dict[str, Any], max_iteraciones: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    """Ejecuta el ciclo ReAct con interacciones simuladas."""
    historial = []
    prompt_completo = prompt_base + "\n\n" + problema["descripcion"] + "\n\nPensamiento:"
    
    respuesta_final = ""
    
    for i in range(max_iteraciones):
        print(f"\n--- Iteración {i+1}/{max_iteraciones} ---")
        
        # Obtener el siguiente pensamiento y acción del modelo
        respuesta, error = llamar_lmstudio_api(prompt_completo, temperatura=0.7)
        
        if error:
            print(f"Error en la iteración {i+1}: {error}")
            respuesta_final = "Error en el proceso de ReAct: " + error
            break
        
        # Procesar la respuesta para extraer Pensamiento, Acción, etc.
        reacciones = procesar_reacciones("Pensamiento:" + respuesta)
        
        # Extraer el último pensamiento y acción si existen
        ultimo_pensamiento = reacciones["pensamiento"][-1] if reacciones["pensamiento"] else ""
        ultima_accion = reacciones["accion"][-1] if reacciones["accion"] else ""
        
        print(f"Pensamiento: {ultimo_pensamiento[:100]}..." if len(ultimo_pensamiento) > 100 else f"Pensamiento: {ultimo_pensamiento}")
        print(f"Acción: {ultima_accion[:100]}..." if len(ultima_accion) > 100 else f"Acción: {ultima_accion}")
        
        # Registrar el ciclo actual
        ciclo_actual = {
            "iteracion": i+1,
            "pensamiento": ultimo_pensamiento,
            "accion": ultima_accion,
            "respuesta_completa": respuesta
        }
        
        # Si no hay acción, considerar que hemos terminado
        if not ultima_accion:
            print("No se detectó una acción clara. Intentando continuar...")
            observacion = "No se pudo determinar una acción clara. Por favor, proporciona una acción específica."
        elif "respuesta final" in ultima_accion.lower():
            print("Se encontró una respuesta final.")
            respuesta_final = ultimo_pensamiento + "\n\n" + ultima_accion
            historial.append(ciclo_actual)
            break
        else:
            # Simular una observación basada en la acción
            observacion = simular_observacion(ultima_accion, problema)
        
        print(f"Observación: {observacion}")
        
        # Actualizar el ciclo actual con la observación
        ciclo_actual["observacion"] = observacion
        historial.append(ciclo_actual)
        
        # Actualizar el prompt para la siguiente iteración
        prompt_completo += f"\n{respuesta}\nObservación: {observacion}\n\nPensamiento:"
        
        # Pequeña pausa entre iteraciones
        time.sleep(1)
    
    # Si llegamos al máximo de iteraciones sin respuesta final
    if i == max_iteraciones - 1 and not respuesta_final:
        respuesta_final = "Se alcanzó el máximo de iteraciones sin llegar a una respuesta definitiva."
    
    return respuesta_final, historial

def crear_prompt_react() -> str:
    """Crea un prompt base para la técnica ReAct."""
    return """# Instrucciones para el método ReAct (Reasoning + Acting)

Tu tarea es resolver problemas utilizando la metodología ReAct, que alterna entre razonamiento y acciones.
Debes seguir este formato específico:

1. Pensamiento: Reflexiona sobre el problema, analiza la información disponible y planifica tus próximos pasos.
2. Acción: Especifica una acción concreta para obtener información o resolver parte del problema.
3. Observación: Recibirás el resultado de tu acción (esto lo proporcionaré yo).

Continúa este ciclo hasta resolver completamente el problema. Cuando tengas la respuesta final, indica "Acción: Respuesta final: [tu respuesta]".

## Ejemplos de acciones posibles:
- Buscar información específica
- Calcular una operación matemática
- Analizar datos proporcionados
- Desglosar un problema complejo en partes más simples

## Ejemplo:

Problema: ¿Cuál es la capital de Francia y su población aproximada?

Pensamiento: Necesito determinar cuál es la capital de Francia y luego averiguar su población aproximada.
Acción: Buscar cuál es la capital de Francia.
Observación: La capital de Francia es París.

Pensamiento: Ahora que sé que la capital es París, necesito averiguar su población aproximada.
Acción: Buscar la población actual de París.
Observación: Según datos recientes, la población de París es de aproximadamente 2.16 millones en la ciudad propiamente dicha, y más de 12 millones en el área metropolitana.

Pensamiento: Ahora tengo toda la información necesaria para responder a la pregunta.
Acción: Respuesta final: La capital de Francia es París, con una población de aproximadamente 2.16 millones de habitantes en la ciudad y más de 12 millones en el área metropolitana.

## Ahora resuelve el siguiente problema usando el método ReAct:"""

def crear_problemas_demo() -> List[Dict[str, Any]]:
    """Crea una lista de problemas de demostración para ReAct."""
    return [
        {
            "id": 1,
            "tipo": "planificacion",
            "descripcion": "Planifica una ruta para escalar el Monte Everest, considerando la preparación necesaria, equipamiento y mejores temporadas.",
            "dificultad": "alta"
        },
        {
            "id": 2,
            "tipo": "problema_matematico",
            "descripcion": "Un triángulo tiene una base de 6 cm y una altura de 8 cm. ¿Cuál es su área y cómo se compara con un círculo de radio 5 cm?",
            "dificultad": "media"
        },
        {
            "id": 3,
            "tipo": "planeacion_viaje",
            "descripcion": "Necesito planificar un viaje de 5 días de Madrid a París con un presupuesto de 1000€. ¿Cuál sería la mejor manera de organizar el transporte y alojamiento?",
            "dificultad": "baja"
        }
    ]

def guardar_resultado(resultado: str, historial: List[Dict[str, Any]], problema_id: int) -> None:
    """Guarda los resultados de la ejecución en archivos."""
    # Crear directorio para resultados si no existe
    if not os.path.exists("resultados_react"):
        os.makedirs("resultados_react")
    
    # Guardar respuesta final
    with open(f"resultados_react/respuesta_final_problema_{problema_id}.txt", "w", encoding="utf-8") as f:
        f.write(resultado)
    
    # Guardar historial completo en formato JSON
    with open(f"resultados_react/historial_problema_{problema_id}.json", "w", encoding="utf-8") as f:
        json.dump(historial, f, indent=2, ensure_ascii=False)
    
    # Crear un informe legible del proceso
    with open(f"resultados_react/informe_problema_{problema_id}.txt", "w", encoding="utf-8") as f:
        f.write(f"# Informe de ejecución ReAct - Problema {problema_id}\n\n")
        for ciclo in historial:
            f.write(f"## Iteración {ciclo['iteracion']}\n\n")
            f.write(f"### Pensamiento\n{ciclo['pensamiento']}\n\n")
            f.write(f"### Acción\n{ciclo['accion']}\n\n")
            if 'observacion' in ciclo:
                f.write(f"### Observación\n{ciclo['observacion']}\n\n")
            f.write("---\n\n")
        f.write(f"\n## Respuesta Final\n{resultado}")

if __name__ == '__main__':
    print("=== Demostración de ReAct con LM Studio ===")
    print("Verificando que LM Studio esté en ejecución...")
    
    if not verificar_lmstudio_disponible():
        print("¡Error! No se pudo conectar con LM Studio.")
        print("Asegúrate de que LM Studio esté en ejecución y que:")
        print("1. Hayas cargado un modelo")
        print("2. Hayas activado el servidor local en la pestaña 'Server'")
        print("3. El servidor esté escuchando en http://localhost:1234")
        exit(1)
    
    print("LM Studio está en ejecución y listo para usar.")
    
    # Crear prompt base para ReAct
    prompt_base = crear_prompt_react()
    
    # Obtener lista de problemas demo
    problemas = crear_problemas_demo()
    
    # Mostrar problemas disponibles
    print("\nProblemas disponibles para demostración:")
    for problema in problemas:
        print(f"{problema['id']}. {problema['descripcion']} (Dificultad: {problema['dificultad']})")
    
    # Permitir selección del problema o usar uno por defecto
    seleccion = input("\nSelecciona un número de problema (o presiona Enter para usar el primero): ").strip()
    problema_seleccionado = None
    
    if seleccion and seleccion.isdigit():
        id_seleccionado = int(seleccion)
        for problema in problemas:
            if problema["id"] == id_seleccionado:
                problema_seleccionado = problema
                break
    
    if not problema_seleccionado:
        problema_seleccionado = problemas[0]
    
    print(f"\nResolviendo problema: {problema_seleccionado['descripcion']}")
    
    # Ajustes para LM Studio
    print("\nAjustando configuración para LM Studio...")
    
    # Número máximo de iteraciones
    max_iteraciones = 5
    print(f"Máximo de iteraciones configurado: {max_iteraciones}")
    
    # Ejecutar ReAct
    print("\nIniciando proceso ReAct...")
    respuesta_final, historial = ejecutar_react(
        prompt_base, 
        problema_seleccionado, 
        max_iteraciones
    )
    
    # Mostrar resultado final
    print("\n=== Resultado Final ===")
    print(respuesta_final)
    
    # Guardar resultados
    print("\nGuardando resultados...")
    guardar_resultado(respuesta_final, historial, problema_seleccionado["id"])
    
    print(f"\nResultados guardados en la carpeta 'resultados_react'")
    print("¡Demostración de ReAct completada!")