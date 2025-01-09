import os
import numpy as np
import cv2
from tkinter import Tk, Label, Button, filedialog, messagebox, simpledialog, Toplevel, Entry, StringVar
import matplotlib.pyplot as plt

def aplicar_transformacion(imagen, matriz_transformacion):
    """
    Aplica una transformación afín a la imagen utilizando una matriz de transformación.

    Parámetros:
        imagen (numpy.ndarray): Imagen de entrada a transformar.
        matriz_transformacion (numpy.ndarray): Matriz de transformación afín (3x3).

    Retorna:
        numpy.ndarray: Imagen transformada.
    """
    filas, columnas = imagen.shape[:2]
    imagen_transformada = np.zeros_like(imagen)
    matriz_inversa = np.linalg.inv(matriz_transformacion)

    for i in range(filas):
        for j in range(columnas):
            transformado = np.dot(matriz_inversa, [j, i, 1])
            x_original, y_original = int(transformado[0]), int(transformado[1])
            if 0 <= x_original < columnas and 0 <= y_original < filas:
                imagen_transformada[i, j] = imagen[y_original, x_original]

    return imagen_transformada

def rotar(angulo, ancho, alto):
    """
    Crea una matriz de rotación respecto al centro de la imagen.

    Parámetros:
        angulo (float): Ángulo de rotación en grados.
        ancho (int): Ancho de la imagen.
        alto (int): Alto de la imagen.

    Retorna:
        numpy.ndarray: Matriz de rotación afín (3x3).
    """
    radianes = np.radians(angulo)
    cos_theta, sin_theta = np.cos(radianes), np.sin(radianes)
    matriz_rotacion = np.array([
        [cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    centro_x, centro_y = ancho // 2, alto // 2
    matriz_traslado_centro = np.array([
        [1, 0, -centro_x],
        [0, 1, -centro_y],
        [0, 0, 1]
    ])
    matriz_traslado_origen = np.array([
        [1, 0, centro_x],
        [0, 1, centro_y],
        [0, 0, 1]
    ])
    return np.dot(np.dot(matriz_traslado_origen, matriz_rotacion), matriz_traslado_centro)

def escalar(factor_x, factor_y, ancho, alto):
    """
    Crea una matriz de escalado respecto al centro de la imagen.

    Parámetros:
        factor_x (float): Factor de escalado en el eje X.
        factor_y (float): Factor de escalado en el eje Y.
        ancho (int): Ancho de la imagen.
        alto (int): Alto de la imagen.

    Retorna:
        numpy.ndarray: Matriz de escalado afín (3x3).
    """
    matriz_escalado = np.array([
        [factor_x, 0, 0],
        [0, factor_y, 0],
        [0, 0, 1]
    ])

    centro_x, centro_y = ancho // 2, alto // 2
    matriz_traslado_centro = np.array([
        [1, 0, -centro_x],
        [0, 1, -centro_y],
        [0, 0, 1]
    ])
    matriz_traslado_origen = np.array([
        [1, 0, centro_x],
        [0, 1, centro_y],
        [0, 0, 1]
    ])

    return np.dot(np.dot(matriz_traslado_origen, matriz_escalado), matriz_traslado_centro)

def reflejar(eje, ancho, alto):
    """
    Crea una matriz de reflexión respecto al centro de la imagen.

    Parámetros:
        eje (str): Eje de reflexión ('horizontal' o 'vertical').
        ancho (int): Ancho de la imagen.
        alto (int): Alto de la imagen.

    Retorna:
        numpy.ndarray: Matriz de reflexión afín (3x3).

    Excepciones:
        ValueError: Si el eje proporcionado no es 'horizontal' o 'vertical'.
    """
    if eje == 'horizontal':
        matriz_reflexion = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    elif eje == 'vertical':
        matriz_reflexion = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Eje no válido. Usa 'horizontal' o 'vertical'.")

    centro_x, centro_y = ancho // 2, alto // 2
    matriz_traslado_centro = np.array([
        [1, 0, -centro_x],
        [0, 1, -centro_y],
        [0, 0, 1]
    ])
    matriz_traslado_origen = np.array([
        [1, 0, centro_x],
        [0, 1, centro_y],
        [0, 0, 1]
    ])

    return np.dot(np.dot(matriz_traslado_origen, matriz_reflexion), matriz_traslado_centro)


def trasladar(dx, dy):
    """
    Crea una matriz de traslación.

    Parámetros:
        dx (float): Desplazamiento en el eje X.
        dy (float): Desplazamiento en el eje Y.

    Retorna:
        numpy.ndarray: Matriz de traslación afín (3x3).
    """
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])


def iniciar_procesamiento():
    """
    Inicia el proceso de transformación de imágenes seleccionadas.

    Verifica si el usuario ha seleccionado imágenes. Si no se han seleccionado, muestra un cuadro de advertencia.
    Si hay imágenes seleccionadas, muestra un cuadro de información con instrucciones sobre cómo proceder.

    Excepciones:
        Muestra una advertencia si no hay imágenes seleccionadas.

    No retorna ningún valor.
    """
    if not imagenes_seleccionadas:
        messagebox.showwarning("Sin imágenes", "Por favor, selecciona imágenes primero.")
        return

    messagebox.showinfo(
        "Instrucciones", 
        "Selecciona una transformación (Rotar, Escalar, Reflejar o Trasladar) para continuar."
    )

        
def mostrar_imagenes(imagen_original, imagen_transformada):
    """
    Muestra la imagen original y la imagen transformada en una ventana de visualización.

    Utiliza matplotlib para mostrar ambas imágenes lado a lado, facilitando la comparación
    entre la imagen original y la transformada.

    Parámetros:
        imagen_original (numpy.ndarray): Imagen original antes de aplicar la transformación.
        imagen_transformada (numpy.ndarray): Imagen resultante después de aplicar la transformación.

    No retorna ningún valor.
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB))
    plt.title("Imagen Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(imagen_transformada, cv2.COLOR_BGR2RGB))
    plt.title("Imagen Transformada")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def procesar_imagenes(imagenes, tipo_transformacion, **parametros):
    """
    Aplica una transformación seleccionada a una lista de imágenes y guarda los resultados.

    Crea directorios para almacenar las imágenes procesadas según el tipo de transformación.
    Carga cada imagen, aplica la transformación especificada y guarda la imagen resultante
    en el directorio correspondiente.

    Parámetros:
        imagenes (list): Lista de rutas de las imágenes a procesar.
        tipo_transformacion (str): Tipo de transformación a aplicar. Puede ser 'rotar', 'escalar',
                                   'reflejar' o 'trasladar'.
        **parametros: Parámetros adicionales necesarios según el tipo de transformación:
            - Para 'rotar': angulo (float) - Ángulo de rotación en grados.
            - Para 'escalar': factor_x (float), factor_y (float) - Factores de escala en X e Y.
            - Para 'reflejar': eje (str) - Eje de reflexión ('horizontal' o 'vertical').
            - Para 'trasladar': dx (float), dy (float) - Desplazamientos en X e Y.

    Excepciones:
        ValueError: Si el tipo de transformación no es válido.

    No retorna ningún valor.
    """
    upload_path = os.path.join(os.getcwd(), "processed")
    os.makedirs(upload_path, exist_ok=True)

    carpeta_tipo = os.path.join(upload_path, tipo_transformacion)
    os.makedirs(carpeta_tipo, exist_ok=True)

    for ruta_entrada in imagenes:
        imagen_original = cv2.imread(ruta_entrada)
        if imagen_original is None:
            print(f"No se pudo cargar la imagen desde {ruta_entrada}.")
            continue

        filas, columnas = imagen_original.shape[:2]

        if tipo_transformacion == "rotar":
            angulo = parametros['angulo']
            matriz = rotar(angulo, columnas, filas)
        elif tipo_transformacion == "escalar":
            factor_x = parametros['factor_x']
            factor_y = parametros['factor_y']
            matriz = escalar(factor_x, factor_y, columnas, filas)
        elif tipo_transformacion == "reflejar":
            eje = parametros['eje']
            matriz = reflejar(eje, columnas, filas)
        elif tipo_transformacion == "trasladar":
            dx = parametros['dx']
            dy = parametros['dy']
            matriz = trasladar(dx, dy)
        else:
            raise ValueError("Transformación no válida")

        imagen_transformada = aplicar_transformacion(imagen_original, matriz)

        mostrar_imagenes(imagen_original, imagen_transformada)

        nombre_archivo = os.path.basename(ruta_entrada)
        output_path = os.path.join(carpeta_tipo, nombre_archivo)
        cv2.imwrite(output_path, imagen_transformada)
        print(f"Procesada y guardada en: {output_path}")


def seleccionar_imagenes():
    """
    Abre un cuadro de diálogo para que el usuario seleccione imágenes y almacena las rutas seleccionadas.

    Utiliza un cuadro de diálogo de archivo para permitir al usuario seleccionar múltiples imágenes.
    Si se seleccionan imágenes, muestra un mensaje informando la cantidad de imágenes cargadas.
    Si no se seleccionan imágenes, muestra una advertencia.

    No recibe parámetros.

    Variables globales:
        imagenes_seleccionadas (list): Almacena las rutas de las imágenes seleccionadas por el usuario.

    No retorna ningún valor.
    """
    global imagenes_seleccionadas
    imagenes_seleccionadas = filedialog.askopenfilenames(
        title="Seleccionar imágenes",
        filetypes=[("Imágenes", "*.jpeg;*.jpg;*.png")]
    )
    if imagenes_seleccionadas:
        messagebox.showinfo("Imágenes seleccionadas", f"{len(imagenes_seleccionadas)} imágenes cargadas.")
    else:
        messagebox.showwarning("Sin selección", "No se seleccionaron imágenes.")




def iniciar_procesamiento():
    """
    Solicita al usuario el tipo de transformación a aplicar y procesa las imágenes seleccionadas.

    Verifica si hay imágenes seleccionadas antes de continuar. Si no hay imágenes, muestra una advertencia.
    Solicita al usuario que elija el tipo de transformación (rotar, escalar, reflejar o trasladar) y los
    parámetros necesarios según la transformación seleccionada. Luego, llama a la función `procesar_imagenes`
    para aplicar la transformación.

    Excepciones:
        Muestra un cuadro de error si el tipo de transformación no es válido.
        Muestra un cuadro de advertencia si no hay imágenes seleccionadas.

    No recibe parámetros.

    No retorna ningún valor.
    """
    if not imagenes_seleccionadas:
        messagebox.showwarning("Sin imágenes", "Por favor, selecciona imágenes primero.")
        return

    tipo = simpledialog.askstring("Transformación", "Ingresa el tipo de transformación (rotar, escalar, reflejar, trasladar):")
    if tipo == "rotar":
        angulo = float(simpledialog.askstring("Ángulo", "Ingresa el ángulo de rotación (grados):"))
        procesar_imagenes(imagenes_seleccionadas, tipo_transformacion="rotar", angulo=angulo)
    elif tipo == "escalar":
        factor_x = float(simpledialog.askstring("Escala X", "Ingresa el factor de escalado en X:"))
        factor_y = float(simpledialog.askstring("Escala Y", "Ingresa el factor de escalado en Y:"))
        procesar_imagenes(imagenes_seleccionadas, tipo_transformacion="escalar", factor_x=factor_x, factor_y=factor_y)
    elif tipo == "reflejar":
        eje = simpledialog.askstring("Reflejo", "Ingresa el eje de reflexión ('horizontal' o 'vertical'):")
        procesar_imagenes(imagenes_seleccionadas, tipo_transformacion="reflejar", eje=eje)
    elif tipo == "trasladar":
        dx = float(simpledialog.askstring("Desplazamiento X", "Ingresa el desplazamiento en X:"))
        dy = float(simpledialog.askstring("Desplazamiento Y", "Ingresa el desplazamiento en Y:"))
        procesar_imagenes(imagenes_seleccionadas, tipo_transformacion="trasladar", dx=dx, dy=dy)
    else:
        messagebox.showerror("Error", "Tipo de transformación no reconocido.")
        return

    messagebox.showinfo("Procesamiento completo", "Las imágenes se han procesado.")


def abrir_rotar():
    """
    Crea una ventana emergente para ingresar el ángulo de rotación de una imagen.

    La ventana incluye una entrada de texto para el ángulo y un botón para aplicar la rotación.
    Al presionar el botón, se llama a la función `aplicar_rotar` para procesar la imagen seleccionada.

    No recibe parámetros.
    
    No retorna ningún valor.
    """ 
    ventana = Toplevel()
    ventana.title("Rotar Imagen")
    Label(ventana, text="Ángulo de rotación (grados):").pack()
    angulo = StringVar()
    Entry(ventana, textvariable=angulo).pack()
    Button(ventana, text="Aplicar", command=lambda: aplicar_rotar(ventana, angulo)).pack()

def aplicar_rotar(ventana, angulo):
    """
    Aplica la transformación de rotación a las imágenes seleccionadas según el ángulo proporcionado.

    Parámetros:
        ventana (tkinter.Toplevel): Ventana emergente que será cerrada después de aplicar la rotación.
        angulo (tkinter.StringVar): Variable de tipo cadena que contiene el ángulo de rotación ingresado por el usuario.

    Excepciones:
        ValueError: Muestra un cuadro de error si el valor ingresado no es numérico.

    No retorna ningún valor.
    """
    try:
        angulo_valor = float(angulo.get())
        procesar_imagenes(imagenes_seleccionadas, tipo_transformacion="rotar", angulo=angulo_valor)
        ventana.destroy()
    except ValueError:
        messagebox.showerror("Error", "Por favor, ingresa un valor numérico para el ángulo.")

def abrir_escalar():
    """
    Crea una ventana emergente para ingresar los factores de escala en los ejes X e Y.

    La ventana incluye entradas de texto para los factores de escala y un botón para aplicar la transformación.
    Al presionar el botón, se llama a la función `aplicar_escalar` para procesar la imagen seleccionada.

    No recibe parámetros.
    
    No retorna ningún valor.
    """
    ventana = Toplevel()
    ventana.title("Escalar Imagen")
    Label(ventana, text="Factor de escala en X:").pack()
    factor_x = StringVar()
    Entry(ventana, textvariable=factor_x).pack()
    Label(ventana, text="Factor de escala en Y:").pack()
    factor_y = StringVar()
    Entry(ventana, textvariable=factor_y).pack()
    Button(ventana, text="Aplicar", command=lambda: aplicar_escalar(ventana, factor_x, factor_y)).pack()

def aplicar_escalar(ventana, factor_x, factor_y):
    """
    Aplica la transformación de escalado a las imágenes seleccionadas según los factores proporcionados.

    Parámetros:
        ventana (tkinter.Toplevel): Ventana emergente que será cerrada después de aplicar la transformación.
        factor_x (tkinter.StringVar): Factor de escala en el eje X ingresado por el usuario.
        factor_y (tkinter.StringVar): Factor de escala en el eje Y ingresado por el usuario.

    Excepciones:
        ValueError: Muestra un cuadro de error si los valores ingresados no son numéricos.

    No retorna ningún valor.
    """
    try:
        fx = float(factor_x.get())
        fy = float(factor_y.get())
        procesar_imagenes(imagenes_seleccionadas, tipo_transformacion="escalar", factor_x=fx, factor_y=fy)
        ventana.destroy()
    except ValueError:
        messagebox.showerror("Error", "Por favor, ingresa valores numéricos para los factores de escala.")

def abrir_reflejar():
    """
    Crea una ventana emergente para seleccionar el eje de reflexión ('horizontal' o 'vertical').

    La ventana incluye una entrada de texto para el eje y un botón para aplicar la reflexión.
    Al presionar el botón, se llama a la función `aplicar_reflejar` para procesar la imagen seleccionada.

    No recibe parámetros.
    
    No retorna ningún valor.
    """
    ventana = Toplevel()
    ventana.title("Reflejar Imagen")
    Label(ventana, text="Eje de reflexión ('horizontal' o 'vertical')").pack()
    eje = StringVar()
    Entry(ventana, textvariable=eje).pack()
    Button(ventana, text="Aplicar", command=lambda: aplicar_reflejar(ventana, eje)).pack()

def aplicar_reflejar(ventana, eje):
    """
    Aplica la transformación de reflexión a las imágenes seleccionadas según el eje proporcionado.

    Parámetros:
        ventana (tkinter.Toplevel): Ventana emergente que será cerrada después de aplicar la reflexión.
        eje (tkinter.StringVar): Eje de reflexión ingresado por el usuario ('horizontal' o 'vertical').

    Excepciones:
        ValueError: Muestra un cuadro de error si el eje ingresado no es válido.

    No retorna ningún valor.
    """
    eje_valor = eje.get().strip().lower()
    if eje_valor in ["horizontal", "vertical"]:
        procesar_imagenes(imagenes_seleccionadas, tipo_transformacion="reflejar", eje=eje_valor)
        ventana.destroy()
    else:
        messagebox.showerror("Error", "Eje no válido. Usa 'horizontal' o 'vertical'.")

def abrir_trasladar():
    """
    Crea una ventana emergente para ingresar los desplazamientos en los ejes X e Y.

    La ventana incluye entradas de texto para los desplazamientos y un botón para aplicar la traslación.
    Al presionar el botón, se llama a la función `aplicar_trasladar` para procesar la imagen seleccionada.

    No recibe parámetros.
    
    No retorna ningún valor.
    """

    ventana = Toplevel()
    ventana.title("Trasladar Imagen")
    Label(ventana, text="Desplazamiento en X:").pack()
    dx = StringVar()
    Entry(ventana, textvariable=dx).pack()
    Label(ventana, text="Desplazamiento en Y:").pack()
    dy = StringVar()
    Entry(ventana, textvariable=dy).pack()
    Button(ventana, text="Aplicar", command=lambda: aplicar_trasladar(ventana, dx, dy)).pack()

def aplicar_trasladar(ventana, dx, dy):
    """
    Aplica la transformación de traslación a las imágenes seleccionadas según los desplazamientos proporcionados.

    Parámetros:
        ventana (tkinter.Toplevel): Ventana emergente que será cerrada después de aplicar la traslación.
        dx (tkinter.StringVar): Desplazamiento en el eje X ingresado por el usuario.
        dy (tkinter.StringVar): Desplazamiento en el eje Y ingresado por el usuario.

    Excepciones:
        ValueError: Muestra un cuadro de error si los valores ingresados no son numéricos.

    No retorna ningún valor.
    """
    try:
        dx_valor = float(dx.get())
        dy_valor = float(dy.get())
        procesar_imagenes(imagenes_seleccionadas, tipo_transformacion="trasladar", dx=dx_valor, dy=dy_valor)
        ventana.destroy()
    except ValueError:
        messagebox.showerror("Error", "Por favor, ingresa valores numéricos para el desplazamiento.")


imagenes_seleccionadas = []
"""
Variable global que almacena las rutas de las imágenes seleccionadas por el usuario.
"""

def seleccionar_imagenes():
    """
    Abre un cuadro de diálogo para que el usuario seleccione imágenes y almacena las rutas seleccionadas.

    Utiliza el cuadro de diálogo de archivo proporcionado por `filedialog` para permitir al usuario
    seleccionar múltiples imágenes. Si se seleccionan imágenes, muestra un mensaje informando la cantidad
    de imágenes cargadas. Si no se seleccionan imágenes, muestra una advertencia.

    Variables globales:
        imagenes_seleccionadas (list): Almacena las rutas de las imágenes seleccionadas por el usuario.

    No recibe parámetros.
    
    No retorna ningún valor.
    """

    global imagenes_seleccionadas
    imagenes_seleccionadas = filedialog.askopenfilenames(
        title="Seleccionar imágenes",
        filetypes=[("Imágenes", "*.jpeg;*.jpg;*.png")]
    )
    if imagenes_seleccionadas:
        messagebox.showinfo("Imágenes seleccionadas", f"{len(imagenes_seleccionadas)} imágenes cargadas.")
    else:
        messagebox.showwarning("Sin selección", "No se seleccionaron imágenes.")

root = Tk()  # Crear ventana principal
root.title("Editor de Imágenes")  # Establecer título de la ventana
root.geometry("300x300")  # Establecer tamaño de la ventana

# Etiqueta descriptiva
label = Label(root, text="Editor de imágenes\nSelecciona y transforma imágenes fácilmente", wraplength=250, pady=20)
label.pack()  # Colocar la etiqueta en la ventana

# Botón para seleccionar imágenes
boton_cargar = Button(root, text="Seleccionar Imágenes", command=seleccionar_imagenes, bg="lightblue", padx=10, pady=5)
boton_cargar.pack(pady=10)  # Colocar el botón en la ventana

# Botón para abrir la ventana de rotación
boton_rotar = Button(root, text="Rotar", command=abrir_rotar, bg="lightgreen", padx=10, pady=5)
boton_rotar.pack(pady=5)  # Colocar el botón en la ventana

# Botón para abrir la ventana de escalado
boton_escalar = Button(root, text="Escalar", command=abrir_escalar, bg="lightgreen", padx=10, pady=5)
boton_escalar.pack(pady=5)  # Colocar el botón en la ventana

# Botón para abrir la ventana de reflexión
boton_reflejar = Button(root, text="Reflejar", command=abrir_reflejar, bg="lightgreen", padx=10, pady=5)
boton_reflejar.pack(pady=5)  # Colocar el botón en la ventana

# Botón para abrir la ventana de traslación
boton_trasladar = Button(root, text="Trasladar", command=abrir_trasladar, bg="lightgreen", padx=10, pady=5)
boton_trasladar.pack(pady=5)  # Colocar el botón en la ventana

root.mainloop()  # Iniciar el bucle principal de la aplicación
