from collections import deque
from pathlib import Path
import cv2
import numpy as np

video = cv2.VideoCapture(str(Path(__file__).resolve().parent / "videos" / "billar.mp4"))
if not video.isOpened():
    raise SystemExit("Error al abrir el video.")
fps = video.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    raise SystemExit("No se pudieron obtener los FPS.")
pausa_ms = max(1, round(1000 / fps))

ancho_max = 960                                  
intervalo_deteccion = max(1, round(fps * 0.5))   

azul_bajo = np.array([92, 110, 160], dtype=np.uint8)    
azul_alto = np.array([112, 255, 255], dtype=np.uint8)
umbral_oscuro = 60       
area_min = 0.0003        
margen_slot = 0.08       

tol_h, tol_s, brillo = 12, 70, 75   
area_bola_min, area_bola_max = 0.0003, 0.0040   
blanco_bajo = np.array([0, 0, 200], dtype=np.uint8)
blanco_alto = np.array([180, 60, 255], dtype=np.uint8)
fraccion_blanca_min = 0.7   

tamano_historial, tolerancia_mesa, max_rechazos = 15, 0.04, 20   
radio_tronera, radio_bola = 18, 9
angulo_max_corte = 90    

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_grande = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
fuente = cv2.FONT_HERSHEY_SIMPLEX
historial = deque(maxlen=tamano_historial)


espejos = [([2, 1, 0, 5, 4, 3], -1, 1), ([3, 4, 5, 0, 1, 2], 1, -1), ([5, 4, 3, 2, 1, 0], -1, -1)]


def redimensionar(imagen):
    if imagen.shape[1] <= ancho_max:
        return imagen
    escala = ancho_max / imagen.shape[1]
    return cv2.resize(imagen, (ancho_max, int(imagen.shape[0] * escala)), interpolation=cv2.INTER_AREA)


def rotular(imagen, texto, x, y, color, escala=0.6, grosor=2):
    ancho_texto = cv2.getTextSize(texto, fuente, escala, grosor)[0][0]
    x = max(5, min(x - ancho_texto // 2, imagen.shape[1] - ancho_texto - 5))
    cv2.putText(imagen, texto, (x, max(15, y)), fuente, escala, color, grosor)



def detectar_mesa(fotograma):
    suave = cv2.GaussianBlur(fotograma, (5, 5), 0)
    hsv = cv2.cvtColor(suave, cv2.COLOR_BGR2HSV)

    tapete = cv2.inRange(hsv, azul_bajo, azul_alto)
    tapete = cv2.morphologyEx(tapete, cv2.MORPH_OPEN, kernel_grande)
    tapete = cv2.morphologyEx(tapete, cv2.MORPH_CLOSE, kernel_grande)
    contornos, _ = cv2.findContours(tapete, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    contorno = max(contornos, key=cv2.contourArea)
    relleno = np.zeros_like(tapete)
    cv2.drawContours(relleno, [contorno], -1, 255, cv2.FILLED)   
    mx, my, mw, mh = cv2.boundingRect(contorno)
    cx = mx + mw / 2

    zona_segura = cv2.erode(tapete, kernel_grande)
    if np.count_nonzero(zona_segura) < 200:
        zona_segura = tapete
    color_tapete = np.median(hsv[zona_segura > 0], axis=0)      


    gris = cv2.cvtColor(suave, cv2.COLOR_BGR2GRAY)
    _, oscura = cv2.threshold(gris, umbral_oscuro, 255, cv2.THRESH_BINARY_INV)
    oscura = cv2.morphologyEx(oscura, cv2.MORPH_OPEN, kernel)
    oscura = cv2.morphologyEx(oscura, cv2.MORPH_CLOSE, kernel)


    ideales = [(mx, my), (cx, my), (mx + mw, my), (mx, my + mh), (cx, my + mh), (mx + mw, my + mh)]
    radio = int(margen_slot * min(mw, mh))
    reales = [None] * 6

    for i in (2, 3, 4, 5):
        ix, iy = ideales[i]
        x0, y0 = max(0, int(ix) - radio), max(0, int(iy) - radio)
        ventana = oscura[y0:int(iy) + radio, x0:int(ix) + radio]
        for c in cv2.findContours(ventana, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            m = cv2.moments(c)
            if m["m00"] == 0 or cv2.contourArea(c) < area_min * mw * mh:
                continue
            p = (m["m10"] / m["m00"] + x0, m["m01"] / m["m00"] + y0)
            if reales[i] is None or np.hypot(p[0] - ix, p[1] - iy) < np.hypot(reales[i][0] - ix, reales[i][1] - iy):
                reales[i] = p

    troneras = []
    for i in range(6):
        if reales[i] is not None:
            troneras.append((int(reales[i][0]), int(reales[i][1]), True))
            continue
        estimadas = [(cx + sx * (reales[m[i]][0] - cx), my + mh / 2 + sy * (reales[m[i]][1] - my - mh / 2))
                     for m, sx, sy in espejos if reales[m[i]] is not None]
        ex, ey = np.mean(estimadas, axis=0) if estimadas else ideales[i]
        troneras.append((int(ex), int(ey), False))

    return (mx, my, mw, mh), troneras, relleno, color_tapete


def es_coherente(mesa):

    if len(historial) < 5:
        return True
    mediana = np.median([h[0] for h in historial], axis=0)
    return bool(np.all(np.abs(np.array(mesa, dtype=float) - mediana) <= tolerancia_mesa * max(mediana[2:])))


def mesa_estable():
    rect = np.median([h[0] for h in historial], axis=0)
    puntos = np.median([h[1] for h in historial], axis=0)
    detectadas = np.mean([h[2] for h in historial], axis=0) >= 0.5
    return tuple(int(v) for v in rect), [(int(p[0]), int(p[1]), bool(d)) for p, d in zip(puntos, detectadas)]


def detectar_bolas(fotograma, mesa, relleno, color_tapete, troneras, radio_ref):
    mx, my, mw, mh = mesa
    hsv = cv2.cvtColor(cv2.GaussianBlur(fotograma[my:my + mh, mx:mx + mw], (5, 5), 0), cv2.COLOR_BGR2HSV)
    if hsv.size == 0:
        return []

    h0, s0, v0 = color_tapete
    bajo = np.array([max(0, h0 - tol_h), max(0, s0 - tol_s), v0 * brillo / 100], dtype=np.uint8)
    alto = np.array([min(179, h0 + tol_h), min(255, s0 + tol_s), 255], dtype=np.uint8)
    mascara = cv2.bitwise_and(cv2.bitwise_not(cv2.inRange(hsv, bajo, alto)), relleno[my:my + mh, mx:mx + mw])
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    blanco = cv2.inRange(hsv, blanco_bajo, blanco_alto)


    if radio_ref:
        radio_min, radio_max, umbral_pico = 0.6 * radio_ref, 1.5 * radio_ref, 0.7 * radio_ref
    else:
        radio_min = np.sqrt(area_bola_min * mw * mh / np.pi)
        radio_max = np.sqrt(area_bola_max * mw * mh / np.pi)
        umbral_pico = 0.6 * radio_min


    distancia = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
    picos = cv2.threshold(distancia, umbral_pico, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

    bolas = []
    for contorno in cv2.findContours(picos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        pico = np.zeros(mascara.shape, dtype=np.uint8)
        cv2.drawContours(pico, [contorno], -1, 255, cv2.FILLED)
        radio, centro = cv2.minMaxLoc(distancia, mask=pico)[1::2]
        bx, by = centro[0] + mx, centro[1] + my

        if not radio_min <= radio <= radio_max:
            continue
        if any(np.hypot(bx - tx, by - ty) < 1.2 * radio_tronera for tx, ty, _ in troneras):
            continue

        disco = np.zeros(mascara.shape, dtype=np.uint8)
        cv2.circle(disco, centro, int(radio), 255, -1)
        bolas.append((bx, by, radio, cv2.mean(blanco, mask=disco)[0] / 255))

    return bolas


def elegir_tronera(bola, blanca, troneras, mesa, radio_fisico, obstaculos=None):
    bx, by = bola
    tiro = np.array([bx - blanca[0], by - blanca[1]], dtype=float)
    mx, my, mw, mh = mesa
    mejor, mejor_criterio = None, None

    for tx, ty, _ in troneras:
        hacia = np.array([tx - bx, ty - by], dtype=float)
        dist = np.linalg.norm(hacia)
        if dist == 0 or np.linalg.norm(tiro) == 0:
            continue

        coseno = np.dot(tiro, hacia) / (np.linalg.norm(tiro) * dist)
        angulo = np.degrees(np.arccos(np.clip(coseno, -1, 1)))
        if angulo > angulo_max_corte:
            continue

        fx = bx - hacia[0] / dist * 2 * radio_fisico
        fy = by - hacia[1] / dist * 2 * radio_fisico
        if not (mx + radio_fisico <= fx <= mx + mw - radio_fisico and
                my + radio_fisico <= fy <= my + mh - radio_fisico):
            continue


        criterio = (round(angulo, 2), dist)
        if mejor_criterio is None or criterio < mejor_criterio:
            mejor, mejor_criterio = (tx, ty), criterio

    return mejor


def calcular_punto_contacto(bola, destino, radio_fisico):
    bx, by = bola
    hacia = np.array([destino[0] - bx, destino[1] - by], dtype=float)
    dist = np.linalg.norm(hacia)
    if dist == 0:
        return None
    punto = np.array([bx, by], dtype=float) - hacia / dist * 2 * radio_fisico
    return tuple(np.round(punto).astype(int))


def bola_que_bloquea(inicio, fin, obstaculos, radio_bola_movimiento):
    inicio = np.array(inicio, dtype=float)
    fin = np.array(fin, dtype=float)
    vector = fin - inicio
    longitud2 = np.dot(vector, vector)
    if longitud2 == 0:
        return None

    for obstaculo in obstaculos:
        centro = np.array(obstaculo[:2], dtype=float)
        t = np.dot(centro - inicio, vector) / longitud2
        if t < 0 or t > 1:
            continue
        punto = inicio + t * vector
        distancia = np.linalg.norm(centro - punto)
        radio_obstaculo = obstaculo[2] if len(obstaculo) > 2 else radio_bola_movimiento
        if distancia < radio_bola_movimiento + radio_obstaculo:
            return obstaculo
    return None


def dibujar_cruz(imagen, centro, radio, color=(0, 0, 255), grosor=3):
    x, y = int(centro[0]), int(centro[1])
    tamano = max(7, int(radio * 1.2))
    cv2.line(imagen, (x - tamano, y - tamano), (x + tamano, y + tamano), color, grosor, cv2.LINE_AA)
    cv2.line(imagen, (x - tamano, y + tamano), (x + tamano, y - tamano), color, grosor, cv2.LINE_AA)


def posicion_cruz_antes_del_obstaculo(inicio, obstaculo, separacion=7):
    origen = np.array(inicio, dtype=float)
    centro = np.array(obstaculo[:2], dtype=float)
    vector = origen - centro
    longitud = np.linalg.norm(vector)
    if longitud == 0:
        return tuple(np.round(centro).astype(int))
    direccion = vector / longitud
    radio_obstaculo = obstaculo[2] if len(obstaculo) > 2 else radio_bola
    posicion = centro + direccion * (radio_obstaculo + separacion)
    return tuple(np.round(posicion).astype(int))


def dibujar_linea_discontinua(imagen, inicio, fin, color, grosor, fase):
    a = np.array(inicio, dtype=float)
    b = np.array(fin, dtype=float)
    vector = b - a
    longitud = np.linalg.norm(vector)
    if longitud == 0:
        return

    direccion = vector / longitud
    largo_guion, espacio, paso = 14, 9, 23
    desplazamiento = fase % paso
    posicion = desplazamiento - paso

    while posicion < longitud:
        inicio_guion = max(0, posicion)
        fin_guion = min(longitud, posicion + largo_guion)
        if fin_guion > inicio_guion:
            p1 = tuple(np.round(a + direccion * inicio_guion).astype(int))
            p2 = tuple(np.round(a + direccion * fin_guion).astype(int))
            cv2.line(imagen, p1, p2, color, grosor, cv2.LINE_AA)
        posicion += paso


mouse_pos = (-1, -1)


def registrar_raton(evento, x, y, flags, datos):
    global mouse_pos
    if evento == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)


pausado, contador, rechazos = False, 0, 0
forzar_frame = False
mesa, troneras, relleno, color_tapete = None, [], None, None
radio_ref, ultima_blanca, imagen = None, None, None
blanca, otras = None, []
fase_linea = 0
cv2.namedWindow("Video de billar")
cv2.setMouseCallback("Video de billar", registrar_raton)

while True:
    inicio = cv2.getTickCount()

    if not pausado or forzar_frame:
        if pausado:
            forzar_frame = False
        correcto, fotograma = video.read()
        if not correcto:
            break
        fotograma = redimensionar(fotograma)

        if contador % intervalo_deteccion == 0:
            deteccion = detectar_mesa(fotograma)
            if deteccion is not None:
                coherente = es_coherente(deteccion[0])
                rechazos = 0 if coherente else rechazos + 1
                if coherente or rechazos >= max_rechazos:
                    if not coherente:
                        historial.clear()
                    rechazos = 0
                    historial.append((np.array(deteccion[0], dtype=float),
                                      np.array([t[:2] for t in deteccion[1]], dtype=float),
                                      np.array([t[2] for t in deteccion[1]], dtype=float)))
                    relleno, color_tapete = deteccion[2], deteccion[3]
                if historial:
                    mesa, troneras = mesa_estable()

        blanca, otras = None, []
        if mesa is not None and relleno is not None:
            bolas = detectar_bolas(fotograma, mesa, relleno, color_tapete, troneras, radio_ref)
            if bolas and max(b[3] for b in bolas) >= fraccion_blanca_min:
                blanca = max(bolas, key=lambda b: b[3])
                ultima_blanca = (int(blanca[0]), int(blanca[1]))
                radio_ref = blanca[2] if radio_ref is None else 0.9 * radio_ref + 0.1 * blanca[2]
            otras = [b for b in bolas if b is not blanca]

        if mesa is not None:
            mx, my, mw, mh = mesa
            cv2.rectangle(fotograma, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
            for i, (tx, ty, detectada) in enumerate(troneras, start=1):
                color = (0, 255, 255) if detectada else (0, 165, 255)
                cv2.circle(fotograma, (tx, ty), radio_tronera, color, 2)
                rotular(fotograma, f"Tronera {i}", tx, ty + 40 if ty < my + mh / 2 else ty - 30, color)

            if ultima_blanca is not None:
                radio_fisico = radio_ref or radio_bola
                for bola_objetivo in otras:
                    bx, by, radio_objetivo, _ = bola_objetivo
                    obstaculos = [b for b in otras if b is not bola_objetivo]
                    if blanca is not None:
                        obstaculos.append(blanca)
                    destino = elegir_tronera(
                        (bx, by), ultima_blanca, troneras, mesa, radio_objetivo, obstaculos
                    )
                    if destino is not None:
                        bloqueadora_salida = bola_que_bloquea(
                            (bx, by), destino, obstaculos, radio_objetivo
                        )
                        contacto = calcular_punto_contacto(
                            (bx, by), destino, radio_objetivo
                        )
                        bloqueadora_entrada = None
                        if contacto is not None:
                            obstaculos_entrada = [b for b in otras if b is not bola_objetivo]
                            bloqueadora_entrada = bola_que_bloquea(
                                ultima_blanca, contacto, obstaculos_entrada, radio_fisico
                            )

                        posible = bloqueadora_salida is None and bloqueadora_entrada is None
                        color_linea = (0, 255, 0) if posible else (0, 0, 255)
                        cv2.line(fotograma, (int(bx), int(by)), destino, color_linea, 2, cv2.LINE_AA)

        for bx, by, _, _ in otras:
            cv2.circle(fotograma, (int(bx), int(by)), radio_bola, (255, 255, 0), 2)
            rotular(fotograma, "Bola", int(bx), int(by) - radio_bola - 6, (255, 255, 0), 0.45, 1)
        if blanca is not None:
            cv2.circle(fotograma, ultima_blanca, radio_bola, (255, 0, 255), 2)
            rotular(fotograma, "Bola blanca", ultima_blanca[0], ultima_blanca[1] - radio_bola - 10, (255, 0, 255))

        imagen = fotograma
        contador += 1

    mostrada = imagen.copy()
    if mesa is not None and ultima_blanca is not None:
        mx_raton, my_raton = mouse_pos
        bola_hover = None

        for bola_objetivo in otras:
            bx, by, radio, _ = bola_objetivo
            if np.hypot(mx_raton - bx, my_raton - by) <= max(radio, radio_bola) + 4:
                bola_hover = bola_objetivo
                break

        if bola_hover is not None:
            bx, by, _, _ = bola_hover
            radio_fisico = radio_ref or radio_bola
            obstaculos_objetivo = [b for b in otras if b is not bola_hover]
            if blanca is not None:
                obstaculos_objetivo.append(blanca)
            destino = elegir_tronera(
                (bx, by), ultima_blanca, troneras, mesa, radio_fisico, obstaculos_objetivo
            )

            if destino is not None:
                bloqueadora_tronera = bola_que_bloquea(
                    (bx, by), destino, obstaculos_objetivo, radio_fisico
                )
                contacto = calcular_punto_contacto((bx, by), destino, radio_fisico)
                bloqueadora_entrada = None
                if contacto is not None:
                    obstaculos_entrada = [b for b in otras if b is not bola_hover]
                    bloqueadora_entrada = bola_que_bloquea(
                        ultima_blanca, contacto, obstaculos_entrada, radio_fisico
                    )

                bloqueadora = bloqueadora_entrada or bloqueadora_tronera
                if bloqueadora is not None:
                    inicio_bloqueo = ultima_blanca if bloqueadora_entrada is not None else (bx, by)
                    posicion_cruz = posicion_cruz_antes_del_obstaculo(inicio_bloqueo, bloqueadora)
                    dibujar_cruz(mostrada, posicion_cruz, bloqueadora[2])
                elif contacto is not None:
                    radio_fantasma = max(2, int(radio_fisico))
                    dibujar_linea_discontinua(
                        mostrada, ultima_blanca, contacto, (0, 255, 0), 2, fase_linea
                    )
                    cv2.circle(mostrada, contacto, radio_fantasma, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(mostrada, contacto, 2, (0, 255, 0), -1, cv2.LINE_AA)

    fase_linea = (fase_linea + 2) % 23

    if pausado:
        cv2.putText(mostrada, "PAUSA", (mostrada.shape[1] - 120, 30), fuente, 0.8, (0, 0, 255), 2)
    cv2.imshow("Video de billar", mostrada)

    gastado_ms = (cv2.getTickCount() - inicio) * 1000 / cv2.getTickFrequency()
    tecla = cv2.waitKey(30 if pausado else max(1, int(pausa_ms - gastado_ms))) & 0xFF

    if tecla in (13, 10):      
        if pausado:
            pausado = False
        else:
            pausado = True
            forzar_frame = True  
    elif tecla == ord('r'):    
        historial.clear()
        mesa, troneras, relleno, color_tapete, contador, rechazos = None, [], None, None, 0, 0
    elif tecla == ord('q'):
        break

video.release()
cv2.destroyAllWindows()