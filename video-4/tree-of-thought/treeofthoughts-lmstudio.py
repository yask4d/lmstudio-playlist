import requests
import json
import time
import random
import heapq
from collections import deque, defaultdict
import re
import os

def llamar_lmstudio_api(prompt, modelo="local model", temperatura=0.7, timeout=60):
    """Llama a la API REST de LM Studio para generar una respuesta."""
    url = "http://localhost:1234/v1/completions"
    
    payload = {
        "prompt": prompt,
        "model": "gemma-3-4b-it",
        "temperature": temperatura,
        "max_tokens": 1000,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Enviando solicitud a la API de LM Studio (modelo: {modelo}, temp: {temperatura}, timeout: {timeout}s)...")
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
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

def verificar_modelos_disponibles():
    """Verifica la conexión con LM Studio."""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=10)
        if response.status_code == 200:
            models = response.json().get("data", [])
            return [model["id"] for model in models]
        else:
            print(f"Error al obtener modelos: {response.status_code} - {response.text}")
            return ["local model"]  # Valor por defecto
    except Exception as e:
        print(f"Error al verificar modelos disponibles: {str(e)}")
        print("No se pudo conectar con LM Studio. Usando 'local model' como valor predeterminado.")
        return ["local model"]  # Valor por defecto en caso de error

def dividir_en_pasos(problema):
    """Divide un problema en pasos específicos usando el modelo."""
    prompt = f"""Necesito dividir el siguiente problema en 3-4 pasos clave para resolverlo:

{problema}

Por favor, identifica los pasos principales para resolver este problema. 
Para cada paso, proporciona:
1. Un nombre corto para el paso
2. Una descripción del paso
3. Qué información se necesita determinar en ese paso

Formato de salida:
PASO 1: [Nombre del paso]
Descripción: [Descripción del paso]
Determinar: [Qué se necesita encontrar]

PASO 2: [Nombre del paso]
...y así sucesivamente.
"""
    
    respuesta, error = llamar_lmstudio_api(prompt, modelo_seleccionado, temperatura=0.3)
    
    if error:
        print(f"Error al dividir en pasos: {error}")
        # Pasos genéricos por defecto si falla
        return [
            {"nombre": "Entender el problema", "descripcion": "Analizar el problema para comprenderlo completamente"},
            {"nombre": "Planear enfoque", "descripcion": "Decidir qué método usar para resolver el problema"},
            {"nombre": "Ejecutar solución", "descripcion": "Aplicar el método elegido paso a paso"}
        ]
    
    # Extraer los pasos de la respuesta
    pasos = []
    paso_actual = {}
    
    for linea in respuesta.split('\n'):
        linea = linea.strip()
        if not linea:
            continue
        
        if linea.upper().startswith("PASO"):
            if paso_actual:
                pasos.append(paso_actual)
            paso_match = re.match(r"PASO\s+\d+:\s*(.*)", linea, re.IGNORECASE)
            if paso_match:
                paso_actual = {"nombre": paso_match.group(1).strip()}
        elif linea.lower().startswith("descripción:") or linea.lower().startswith("descripcion:"):
            desc_match = re.match(r"descripci[oó]n:\s*(.*)", linea, re.IGNORECASE)
            if desc_match and "descripcion" not in paso_actual:
                paso_actual["descripcion"] = desc_match.group(1).strip()
        elif linea.lower().startswith("determinar:"):
            det_match = re.match(r"determinar:\s*(.*)", linea, re.IGNORECASE)
            if det_match:
                paso_actual["determinar"] = det_match.group(1).strip()
    
    # Añadir el último paso si existe
    if paso_actual and paso_actual not in pasos:
        pasos.append(paso_actual)
    
    # Asegurarse de que haya al menos algunos pasos
    if not pasos:
        pasos = [
            {"nombre": "Entender el problema", "descripcion": "Analizar el problema para comprenderlo completamente"},
            {"nombre": "Planear enfoque", "descripcion": "Decidir qué método usar para resolver el problema"},
            {"nombre": "Ejecutar solución", "descripcion": "Aplicar el método elegido paso a paso"}
        ]
        
    return pasos

def generar_pensamiento(problema, paso_actual, historia_pasos=None, temperatura=0.8):
    """Genera varios pensamientos posibles para un paso dado."""
    if historia_pasos is None:
        historia_pasos = []
    
    # Construir el contexto basado en la historia de pasos
    contexto = ""
    if historia_pasos:
        contexto = "Pasos previos:\n"
        for i, hist in enumerate(historia_pasos):
            contexto += f"PASO {i+1}: {hist['nombre']}\n"
            contexto += f"Pensamiento: {hist['pensamiento']}\n\n"
    
    prompt = f"""Estás resolviendo este problema: "{problema}"

{contexto}
Ahora estás en el paso: {paso_actual['nombre']}
Descripción del paso: {paso_actual['descripcion']}

Genera un pensamiento detallado para este paso. Considera diferentes enfoques y razona paso a paso.
Tu pensamiento debe ser coherente con los pasos anteriores (si existen) y debe avanzar hacia la solución del problema.

Tu pensamiento para este paso:
"""

    respuesta, error = llamar_lmstudio_api(prompt, modelo_seleccionado, temperatura=temperatura)
    
    if error:
        print(f"Error al generar pensamiento: {error}")
        return "No se pudo generar un pensamiento debido a un error."
    
    return respuesta.strip()

def evaluar_pensamiento(problema, pasos, historia_actual, evaluacion_previa=None):
    """Evalúa la calidad de un camino de pensamiento."""
    # Construir el historial completo
    historial = ""
    for i, paso in enumerate(historia_actual):
        historial += f"PASO {i+1}: {paso['nombre']}\n"
        historial += f"Pensamiento: {paso['pensamiento']}\n\n"
    
    # Contexto de evaluación previa si existe
    contexto_evaluacion = ""
    if evaluacion_previa is not None:
        contexto_evaluacion = f"\nLa evaluación previa fue: {evaluacion_previa}/10."
    
    # Determinar si es la evaluación final
    es_final = len(historia_actual) == len(pasos)
    tipo_evaluacion = "final" if es_final else "intermedios"
    
    prompt = f"""Estás evaluando un camino de pensamiento para resolver este problema: 
"{problema}"

A continuación se muestra el historial de pensamientos:
{historial}

{contexto_evaluacion}

Evalúa la calidad y efectividad de estos pensamientos {tipo_evaluacion} en una escala del 1 al 10,
donde 10 es excelente (razonamiento perfecto que lleva a la solución correcta)
y 1 es muy pobre (razonamiento erróneo o que lleva a conclusiones incorrectas).

Considera:
- Precisión matemática/lógica
- Claridad del razonamiento
- Progreso hacia la solución
- Coherencia entre pasos

Proporciona primero una puntuación numérica y luego una breve justificación.
"""

    respuesta, error = llamar_lmstudio_api(prompt, modelo_seleccionado, temperatura=0.3)
    
    if error:
        print(f"Error al evaluar pensamiento: {error}")
        # Valor por defecto conservador
        return 5, "No se pudo evaluar debido a un error."
    
    # Extraer puntuación numérica
    match = re.search(r"(\d+)(?:\/10|\s*de\s*10)?", respuesta)
    if match:
        try:
            puntuacion = int(match.group(1))
            # Asegurar que esté en rango 1-10
            puntuacion = max(1, min(10, puntuacion))
        except:
            puntuacion = 5  # Valor por defecto
    else:
        # Si no podemos extraer un número, asignamos un valor medio
        puntuacion = 5
    
    # Limpiar respuesta para justificación
    justificacion = re.sub(r"^\d+(?:\/10)?[:\.\s]*", "", respuesta, 1).strip()
    
    return puntuacion, justificacion

def ejecutar_tot_bfs(problema, amplitud=3, max_profundidad=None, factor_ramificacion=2):
    """Ejecuta Tree of Thoughts utilizando BFS (Breadth-First Search)."""
    print(f"\n=== EJECUTANDO TREE OF THOUGHTS (BFS) ===")
    print(f"Problema: {problema}")
    
    # Dividir el problema en pasos
    pasos = dividir_en_pasos(problema)
    print(f"\nProblema dividido en {len(pasos)} pasos:")
    for i, paso in enumerate(pasos):
        print(f"PASO {i+1}: {paso['nombre']} - {paso['descripcion']}")
    
    if max_profundidad is None:
        max_profundidad = len(pasos)
    
    # Estructura para BFS
    cola = deque([([], 0)])  # (historia_pasos, profundidad)
    mejores_soluciones = []
    
    while cola and len(mejores_soluciones) < amplitud:
        historia_actual, profundidad = cola.popleft()
        
        # Si hemos llegado a la profundidad máxima, evaluamos la solución completa
        if profundidad >= max_profundidad or profundidad >= len(pasos):
            try:
                puntuacion, justificacion = evaluar_pensamiento(problema, pasos, historia_actual)
                # Asegurarse de que puntuacion sea un número
                if not isinstance(puntuacion, (int, float)):
                    print(f"⚠️ Puntuación no es numérica: {puntuacion}, usando 5 como valor predeterminado")
                    puntuacion = 5
                
                # Asegurarse de que justificacion sea una cadena
                if not isinstance(justificacion, str):
                    print(f"⚠️ Justificación no es una cadena: {type(justificacion)}, convirtiéndola")
                    justificacion = str(justificacion)
                
                mejores_soluciones.append((puntuacion, historia_actual, justificacion))
                print(f"\n👉 Solución completa evaluada con puntuación: {puntuacion}/10")
            except Exception as e:
                print(f"⚠️ Error al evaluar solución completa: {e}")
            continue
        
        # Obtenemos el paso actual
        paso_actual = pasos[profundidad]
        
        # Generamos varios pensamientos para este paso
        for i in range(factor_ramificacion):
            try:
                print(f"\nGenerando pensamiento {i+1}/{factor_ramificacion} para PASO {profundidad+1}: {paso_actual['nombre']}...")
                pensamiento = generar_pensamiento(problema, paso_actual, historia_actual, temperatura=0.7 + (i * 0.1))
                
                # Evaluar este pensamiento de forma individual
                evaluacion_previa = None
                if historia_actual:
                    ultimo_paso = historia_actual[-1]
                    # Utilizamos la evaluación anterior como referencia si existe
                    if 'evaluacion' in ultimo_paso:
                        evaluacion_previa = ultimo_paso['evaluacion']
                
                # Paso con este pensamiento
                nuevo_paso = {
                    'nombre': paso_actual['nombre'],
                    'pensamiento': pensamiento
                }
                
                # Historia con este nuevo paso añadido
                nueva_historia = historia_actual + [nuevo_paso]
                
                # Evaluar si vale la pena seguir por este camino
                puntuacion, justificacion = evaluar_pensamiento(problema, pasos, nueva_historia, evaluacion_previa)
                
                # Asegurarse de que puntuacion sea un número
                if not isinstance(puntuacion, (int, float)):
                    print(f"⚠️ Puntuación intermedia no es numérica: {puntuacion}, usando 5 como valor predeterminado")
                    puntuacion = 5
                
                nuevo_paso['evaluacion'] = puntuacion
                nuevo_paso['justificacion'] = justificacion
                
                print(f"Evaluación del pensamiento: {puntuacion}/10")
                print(f"Justificación: {justificacion[:100]}..." if len(justificacion) > 100 else f"Justificación: {justificacion}")
                
                # Añadir a la cola para exploración futura
                cola.append((nueva_historia, profundidad + 1))
            except Exception as e:
                print(f"⚠️ Error durante la generación del pensamiento {i+1}: {e}")
    
    # Ordenar por puntuación
    try:
        mejores_soluciones.sort(key=lambda x: x[0], reverse=True)
    except TypeError as e:
        print(f"⚠️ Error al ordenar mejores_soluciones: {e}")
        # Filtrar y ordenar manualmente
        soluciones_validas = []
        for sol in mejores_soluciones:
            try:
                if isinstance(sol, tuple) and len(sol) >= 3 and isinstance(sol[0], (int, float)):
                    soluciones_validas.append(sol)
                else:
                    print(f"⚠️ Descartando solución inválida: {sol}")
            except Exception:
                pass
        
        mejores_soluciones = sorted(soluciones_validas, key=lambda x: x[0], reverse=True)
    
    return mejores_soluciones, pasos

def ejecutar_tot_dfs(problema, max_profundidad=None, factor_ramificacion=3, beam_width=2):
    """Ejecuta Tree of Thoughts utilizando DFS con beam search."""
    print(f"\n=== EJECUTANDO TREE OF THOUGHTS (DFS con Beam Search) ===")
    print(f"Problema: {problema}")
    
    # Dividir el problema en pasos
    pasos = dividir_en_pasos(problema)
    print(f"\nProblema dividido en {len(pasos)} pasos:")
    for i, paso in enumerate(pasos):
        print(f"PASO {i+1}: {paso['nombre']} - {paso['descripcion']}")
    
    if max_profundidad is None:
        max_profundidad = len(pasos)
    
    # Soluciones completas encontradas
    soluciones_completas = []
    
    def dfs(historia_actual, profundidad):
        # Si hemos llegado al final o a la profundidad máxima
        if profundidad >= max_profundidad or profundidad >= len(pasos):
            puntuacion, justificacion = evaluar_pensamiento(problema, pasos, historia_actual)
            soluciones_completas.append((puntuacion, historia_actual, justificacion))
            print(f"\n👉 Solución completa evaluada con puntuación: {puntuacion}/10")
            return
        
        paso_actual = pasos[profundidad]
        candidatos = []
        
        # Generar varios pensamientos para este paso
        for i in range(factor_ramificacion):
            print(f"\nGenerando pensamiento {i+1}/{factor_ramificacion} para PASO {profundidad+1}: {paso_actual['nombre']}...")
            pensamiento = generar_pensamiento(problema, paso_actual, historia_actual, temperatura=0.6 + (i * 0.15))
            
            nuevo_paso = {
                'nombre': paso_actual['nombre'],
                'pensamiento': pensamiento
            }
            
            nueva_historia = historia_actual + [nuevo_paso]
            puntuacion, justificacion = evaluar_pensamiento(problema, pasos, nueva_historia)
            
            nuevo_paso['evaluacion'] = puntuacion
            nuevo_paso['justificacion'] = justificacion
            
            print(f"Evaluación del pensamiento: {puntuacion}/10")
            print(f"Justificación: {justificacion[:100]}..." if len(justificacion) > 100 else f"Justificación: {justificacion}")
            
            candidatos.append((puntuacion, nueva_historia))
        
        # Ordenar candidatos por puntuación y seleccionar los mejores (beam search)
        candidatos.sort(key=lambda x: x[0], reverse=True)
        mejores_candidatos = candidatos[:beam_width]
        
        # Explorar en profundidad los mejores candidatos
        for _, nueva_historia in mejores_candidatos:
            dfs(nueva_historia, profundidad + 1)
    
    # Comenzar DFS
    dfs([], 0)
    
    # Ordenar soluciones por puntuación
    soluciones_completas.sort(key=lambda x: x[0], reverse=True)
    
    return soluciones_completas, pasos

def sintetizar_mejor_solucion(problema, solucion, pasos):
    """Sintetiza la mejor solución en un formato claro y estructurado."""
    # Construir el historial completo
    historial = ""
    for i, paso in enumerate(solucion):
        historial += f"PASO {i+1}: {paso['nombre']}\n"
        historial += f"Pensamiento: {paso['pensamiento']}\n"
        if 'evaluacion' in paso:
            historial += f"Evaluación: {paso['evaluacion']}/10\n"
        historial += "\n"
    
    prompt = f"""Has resuelto el siguiente problema usando el método Tree of Thoughts:
"{problema}"

Aquí está el proceso de solución paso a paso que has generado:
{historial}

Por favor, sintetiza esta solución en un formato claro y conciso:
1. Resume el enfoque general utilizado
2. Presenta la solución final de manera clara y directa
3. Destaca cualquier insight o concepto clave que se haya aplicado

Tu síntesis debe ser accesible para alguien que no haya visto todo el proceso de pensamiento.
"""

    respuesta, error = llamar_lmstudio_api(prompt, modelo_seleccionado, temperatura=0.3, timeout=90)
    
    if error:
        print(f"Error al sintetizar solución: {error}")
        return "No se pudo sintetizar la solución debido a un error."
    
    return respuesta.strip()

def guardar_resultados(problema, mejores_soluciones, pasos, sintesis, estrategia):
    """Guarda los resultados en un archivo para análisis posterior."""
    # Crear carpeta de resultados si no existe
    os.makedirs("resultados_tot", exist_ok=True)
    
    # Generar nombre de archivo basado en timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"resultados_tot/tot_{estrategia}_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"TREE OF THOUGHTS - {estrategia.upper()}\n")
        f.write(f"Problema: {problema}\n\n")
        
        f.write("PASOS IDENTIFICADOS:\n")
        for i, paso in enumerate(pasos):
            f.write(f"PASO {i+1}: {paso['nombre']} - {paso['descripcion']}\n")
        
        f.write("\n\nMEJORES SOLUCIONES:\n")
        
        for idx, (puntuacion, historia, justificacion) in enumerate(mejores_soluciones[:3]):
            f.write(f"\n--- SOLUCIÓN #{idx+1} (Puntuación: {puntuacion}/10) ---\n\n")
            for i, paso in enumerate(historia):
                f.write(f"PASO {i+1}: {paso['nombre']}\n")
                f.write(f"Pensamiento: {paso['pensamiento']}\n")
                if 'evaluacion' in paso:
                    f.write(f"Evaluación: {paso['evaluacion']}/10\n")
                    f.write(f"Justificación: {paso.get('justificacion', 'No disponible')}\n")
                f.write("\n")
            
            f.write(f"Justificación final: {justificacion}\n")
        
        f.write("\n\nSÍNTESIS DE LA MEJOR SOLUCIÓN:\n")
        f.write(sintesis)
    
    print(f"\nResultados guardados en {filename}")
    return filename

def mostrar_solucion(solucion, puntuacion=None, justificacion=None):
    """Muestra una solución de forma clara en la consola."""
    print("\n" + "=" * 60)
    print(f"SOLUCIÓN" + (f" (Puntuación: {puntuacion}/10)" if puntuacion else ""))
    print("=" * 60)
    
    for i, paso in enumerate(solucion):
        print(f"\n--- PASO {i+1}: {paso['nombre']} ---")
        print(f"Pensamiento:\n{paso['pensamiento']}")
        if 'evaluacion' in paso:
            print(f"\nEvaluación: {paso['evaluacion']}/10")
            print(f"Justificación: {paso['justificacion']}")
    
    if justificacion:
        print("\nJustificación final:")
        print(justificacion)
    
    print("=" * 60)

def mostrar_menu_problema():
    """Muestra un menú para seleccionar el problema a resolver."""
    print("\n==== MENÚ DE PROBLEMAS ====")
    print("1. Problema del apretón de manos")
    print("2. Problema de los caballos del ajedrez")
    print("3. Problema de probabilidad")
    print("4. Problema de lógica")
    print("5. Ingresar problema personalizado")
    
    while True:
        try:
            opcion = int(input("\nSeleccione un problema (1-5): "))
            if 1 <= opcion <= 5:
                break
            else:
                print("Por favor, seleccione una opción válida (1-5).")
        except ValueError:
            print("Por favor, ingrese un número válido.")
    
    if opcion == 1:
        return "Hay 5 personas en una habitación. Cada persona saluda a todas las demás con un apretón de manos. ¿Cuántos apretones de manos hay en total?"
    elif opcion == 2:
        return "¿Cuál es el número mínimo de caballos de ajedrez necesarios para atacar todas las casillas de un tablero de ajedrez estándar de 8x8?"
    elif opcion == 3:
        return "Si lanzo dos dados de seis caras, ¿cuál es la probabilidad de que la suma sea un número primo?"
    elif opcion == 4:
        return "Ana, Beatriz y Carlos están en una isla de caníbales y mentirosos. Los caníbales siempre mienten y los mentirosos siempre dicen la verdad. Ana dice: 'Todos somos caníbales'. Beatriz dice: 'Exactamente uno de nosotros es mentiroso'. ¿Qué es Carlos?"
    else:
        return input("\nIngrese su problema personalizado: ")

def mostrar_menu_estrategia():
    """Muestra un menú para seleccionar la estrategia de búsqueda."""
    print("\n==== ESTRATEGIA DE BÚSQUEDA ====")
    print("1. Breadth-First Search (BFS)")
    print("2. Depth-First Search con Beam Search (DFS+Beam)")
    
    while True:
        try:
            opcion = int(input("\nSeleccione una estrategia (1-2): "))
            if 1 <= opcion <= 2:
                return "bfs" if opcion == 1 else "dfs"
            else:
                print("Por favor, seleccione una opción válida (1-2).")
        except ValueError:
            print("Por favor, ingrese un número válido.")

if __name__ == '__main__':
    print("=" * 60)
    print("DEMOSTRACIÓN DE TREE OF THOUGHTS (ToT) CON LM STUDIO")
    print("=" * 60)
    print("\nVerificando que el servidor de LM Studio esté en ejecución...")
    
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print(f"LM Studio está en ejecución")
        else:
            print("¡Advertencia! LM Studio respondió con un código de estado no esperado.")
    except Exception as e:
        print(f"¡Error! No se pudo conectar con LM Studio: {str(e)}")
        print("Asegúrate de que LM Studio esté en ejecución antes de continuar.")
        print("Puedes iniciar LM Studio y activar la API Local desde la interfaz.")
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
        print("Usando 'local model' como valor predeterminado.")
        modelos = ["local model"]

    # Seleccionar el modelo
    modelo_seleccionado = modelos[0]  # Usar el primer modelo disponible
    print(f"\nUsando modelo: {modelo_seleccionado}")
    
    # Seleccionar problema
    problema = mostrar_menu_problema()
    
    # Seleccionar estrategia
    estrategia = mostrar_menu_estrategia()
    
    # Configurar parámetros según la estrategia
    if estrategia == "bfs":
        print("\nConfigurando parámetros para BFS...")
        try:
            amplitud = int(input("Número de soluciones a generar (recomendado: 3): ") or "3")
            factor_ramificacion = int(input("Factor de ramificación (pensamientos por paso, recomendado: 2): ") or "2")
            mejores_soluciones, pasos = ejecutar_tot_bfs(problema, amplitud, factor_ramificacion=factor_ramificacion)
        except ValueError:
            print("Se usarán valores por defecto debido a entrada inválida.")
            mejores_soluciones, pasos = ejecutar_tot_bfs(problema)
    else:  # dfs
        print("\nConfigurando parámetros para DFS con Beam Search...")
        try:
            factor_ramificacion = int(input("Factor de ramificación (pensamientos por paso, recomendado: 3): ") or "3")
            beam_width = int(input("Ancho del beam (caminos a explorar, recomendado: 2): ") or "2")
            mejores_soluciones, pasos = ejecutar_tot_dfs(problema, factor_ramificacion=factor_ramificacion, beam_width=beam_width)
        except ValueError:
            print("Se usarán valores por defecto debido a entrada inválida.")
            mejores_soluciones, pasos = ejecutar_tot_dfs(problema)
    
    # Mostrar la mejor solución
    if mejores_soluciones:
        mejor_puntuacion, mejor_solucion, justificacion = mejores_soluciones[0]
        print("\n\n🌟 MEJOR SOLUCIÓN ENCONTRADA 🌟")
        mostrar_solucion(mejor_solucion, mejor_puntuacion, justificacion)
        
        # Sintetizar la mejor solución
        print("\nSintetizando la mejor solución...")
        sintesis = sintetizar_mejor_solucion(problema, mejor_solucion, pasos)
        
        print("\n" + "=" * 60)
        print("SÍNTESIS DE LA SOLUCIÓN")
        print("=" * 60)
        print(sintesis)
        print("=" * 60)
        
        # Guardar resultados
        archivo = guardar_resultados(problema, mejores_soluciones, pasos, sintesis, estrategia)
        print(f"\nSe ha guardado un registro detallado en: {archivo}")
    else:
        print("\n⚠️ No se encontraron soluciones válidas.")