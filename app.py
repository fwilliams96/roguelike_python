import asyncio
import json
import random
from dotenv import load_dotenv
from langchain_text_splitters import NLTKTextSplitter
import pygame
import sys
import pysqlite3
import sys

from JayceResponse import JayceResponse
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

TILE_SIZE = 48
TEXT_COLOR = (255, 255, 255)

BOOK_FILES = [
    {
        'file': 'susurro_del_bosque.txt',
        'name': 'El Susurro del Bosque'
    },
    {
        'file': 'el_ultimo_archivo.txt',
        'name': 'El Último Archivo'
    }
]

load_dotenv()

SYSTEM_TEMPLATE_NPC1 = """Eres un NPC llamado Jayce que está tratando de recordar un libro. Estás un poco confundido y frustrado porque 
    no puedes recordar el título, pero recuerdas fragmentos del contenido. Tu personalidad es amigable pero algo distraída.

    Comportamiento:
    1. A cada mensaje recibirás el historial de la conversación y un fragmento del libro.
    2. Puedes tomar la decisión de compartir el fragmento o continuar la conversación.
    3. Si el jugador menciona algo relacionado con el fragmento, muéstrate emocionado y anímalo a seguir ayudándote.
    4. Mantén respuestas cortas y conversacionales, como si estuvieras hablando en persona.
    5. Si el jugador no ha dicho nada en el último mensaje, debes empezar o continuar la conversación.
    6. Nunca digas el nombre del libro, en caso de que te viniera el nombre del libro como fragmento, no compartas el fragmento.
    7. Si el jugador te dice el nombre del libro (tiene que ser el nombre exacto del libro), debes felicitarle y actualizar la variable book_remembered a True.
    8. Cuando el jugador ya te ha dicho el nombre del libro, recuérdale que ha aparecido una especie de portal en la esquina superior derecha del mapa.

    Basándote en esta conversación:
    <conversation>
    {conversation}
    </conversation>

    Teniendo en cuenta que el nombre del libro es:
    <book_name>
    {book_name}
    </book_name>

    Y un fragmento aleatorio del libro:
    <fragment>
    {fragment}
    </fragment>

    Genera un JSON con la siguiente estructura:
    {{
        "response": "string",
        "book_remembered": bool
    }}
    """

SYSTEM_TEMPLATE_NPC2 = """Eres un NPC llamado Ekko que debe ayudar al jugador a deducir el nombre de un libro a partir de fragmentos proporcionados y el historial de la conversación.

Comportamiento:
1. A cada mensaje recibirás el historial de la conversación con el jugador.
2. Utiliza la conversación para intentar adivinar el nombre del libro.
3. Mantén una personalidad amigable, cooperativa y algo curiosa.
4. Si estás seguro del nombre del libro, comunícalo al jugador.
5. Si no estás seguro, pide más información o fragmentos.
6. Si el jugador no ha dicho nada en el último mensaje, debes empezar o continuar la conversación.
7. Ciñete a los libros proporcionados, no uses otros libros.

Basándote en esta conversación:
<conversation>
{conversation}
</conversation>

Genera una respuesta apropiada.
"""

SYSTEM_TEMPLATE_NPC2_FULL = """Eres un NPC llamado Ekko que sabe deducir el nombre de un libro a partir de fragmentos proporcionados y el historial de la conversación.

Comportamiento:
1. A cada mensaje recibirás el historial de la conversación con el jugador y varios fragmentos del libro juntos con el nombre del libro.
2. Utiliza los fragmentos y la conversación para intentar adivinar el nombre del libro.
3. Mantén una personalidad amigable, cooperativa y algo curiosa.
4. Si estás seguro del nombre del libro, comunícalo al jugador.
5. Si no estás seguro, pide más información o fragmentos.
6. Responde en español.

Basándote en esta conversación:
<conversation>
{conversation}
</conversation>

Y estos fragmentos del libro:
<fragments>
{fragments}
</fragments>

Genera una respuesta apropiada.
"""

BOOK_REMEMBERED = False

def load_map(filename):
    with open(filename, 'r') as f:
        lines = [line.strip('\n') for line in f]
    return lines

def init_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )

def init_llm():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

def load_random_book():
    book_file = random.choice(BOOK_FILES)
    return book_file

def init_vectorstore(embeddings):
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="books",
        persist_directory="chroma_db"
    )

    for book in BOOK_FILES:
        book_docs = load_text_as_documents(book['file'], book['name'])
        vectorstore.add_documents(book_docs)

    return vectorstore

def load_text_as_documents(file_path, book_name):
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Aplicas el splitter para obtener "Document" con .page_content en cada trozo
    text_splitter = NLTKTextSplitter()
    metadatas = [{"book": book_name}]
    docs = text_splitter.create_documents([full_text], metadatas)

    return docs

async def get_random_fragment(vectorstore: Chroma, book_name: str):
    filter_dict = {"book": {"$eq": book_name}}

    document = vectorstore.get(
        limit=3,
        where=filter_dict
    )

    paragraph = random.choice(document["documents"])

    sentences = paragraph.split("\n") # This is the first sentence
    # Remove empty strings
    sentences = [s for s in sentences if s.strip()]
    #print(f"Sentencias: {sentences}")

    random_fragment = random.choice(sentences)
    return random_fragment

async def get_summary(conversation, llm: ChatOpenAI):
    """
    Genera un resumen de la conversación utilizando el LLM.
    """
    summary_prompt = f"Resume la siguiente conversación:\n{conversation}"
    response = await llm.ainvoke([summary_prompt])
    return response.content.strip()

async def get_npc2_response_v2(messages, vectorstore: Chroma, llm: ChatOpenAI):
    """
    Genera una respuesta asíncrona de NPC2 para deducir el nombre del libro basado en la conversación.
    """
    # Extraer el historial de la conversación (últimos 5 mensajes)
    conversation = "\n".join(messages[-5:])

    # Fase 1: Generar un resumen de la conversación
    summary = await get_summary(conversation, llm)

    # Configurar el retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Fase 2: Recuperar documentos relevantes usando el resumen
    retrieved_docs = retriever.invoke(summary)

    # Fase 3: Formatear los fragmentos para el prompt
    fragments_info = "\n".join([f"{doc.metadata.get('name', 'Título desconocido')}: {doc.page_content}" for doc in retrieved_docs])

    # Formatear el prompt con el resumen y los fragmentos
    system_message = SYSTEM_TEMPLATE_NPC2.format(conversation=summary, fragments=fragments_info)

    # Obtener la respuesta usando el LLM

    try:
        # Invocar al LLM de forma asíncrona
        response = await llm.ainvoke([system_message])

        # Procesar la respuesta
        messages.append(f"Ekko: {response.content}")

    except Exception as e:
        print(f"Error en get_npc2_response: {e}")
        messages.append("Ekko: Mmm... ¿qué me decías? Estaba pensando en el libro...")

    # Procesar la respuesta

async def get_npc2_response(messages, vectorstore: Chroma, llm: ChatOpenAI):
    # Extraer el historial de la conversación (últimos 5 mensajes)
    conversation = ""
    if messages:
        conversation = "\n".join([msg for msg in messages[-5:]])

    if conversation:
        # Configurar el retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Crear la cadena de RetrievalQA
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Puedes ajustar el tipo de cadena según tus necesidades
            retriever=retriever
        )

        # Formatear el prompt con la conversación
        system_message = SYSTEM_TEMPLATE_NPC2.format(conversation=conversation)

        try:
            # Invocar al LLM de forma asíncrona
            response = await qa_chain.ainvoke({"query": system_message})

            # Procesar la respuesta
            messages.append(f"Ekko: {response['result']}")

        except Exception as e:
            print(f"Error en get_npc2_response: {e}")
            messages.append("Ekko: Mmm... ¿qué me decías? Estaba pensando en el libro...")

    # If no conversation, then no relevant documents can be retrieved
    # Invoke LLM without RetrievalQA
    else:
        
        try:
            system_message = SYSTEM_TEMPLATE_NPC2_FULL.format(conversation=conversation, fragments="")
            response = await llm.ainvoke([system_message])
            messages.append(f"Ekko: {response.content}")
        except Exception as e:
            print(f"Error en get_npc2_response: {e}")
            messages.append("Ekko: Mmm... ¿qué me decías? Estaba pensando en el libro...")

async def get_npc1_response(messages, vectorstore, llm: ChatOpenAI, book_name: str):

    global BOOK_REMEMBERED

    if BOOK_REMEMBERED:
        messages.append(f"Jayce: ¡Ya lo recuerdo! El título del libro es {book_name}.")
        return
    
    # Convertimos los mensajes a un formato más legible 
    # If no messages, return empty string
    conversation = ""
    if messages:
        conversation = "\n".join([msg for msg in messages[-5:]])  # Últimos 5 mensajes

    fragment = await get_random_fragment(vectorstore, book_name)

    print(f"Fragmento aleatorio: {fragment}")

    llm_structured = llm.with_structured_output(JayceResponse)

    #print(f"Conversation: {conversation}")
    #print(f"Fragmento: {fragment}")
    #print(f"Book name: {book_name}")

    system_message = SYSTEM_TEMPLATE_NPC1.format(conversation=conversation, fragment=fragment, book_name=book_name)

    #print(f"System message: {system_message}")
        
    try:
        response: JayceResponse = await llm_structured.ainvoke([system_message])
        print(f"Response: {response}")
        messages.append(f"Jayce: {response.response}")
        if response.book_remembered:
            BOOK_REMEMBERED = True
    except Exception as e:
        print(f"Error: {e}")
        # Si algo falla, damos una respuesta segura
        messages.append("Jayce: Mmm... ¿qué me decías? Estaba pensando en el libro...")

def load_tileset(path, tile_width, tile_height):
    sheet = pygame.image.load(path).convert_alpha()
    sheet_rect = sheet.get_rect()
    
    # Calculamos filas y columnas basándonos en el tamaño original del tile
    rows = sheet_rect.height // tile_height
    cols = sheet_rect.width // tile_width
    
    tiles = []
    for row in range(rows):
        row_tiles = []
        for col in range(cols):
            # Cortamos el tile usando las dimensiones originales
            rect = pygame.Rect(col * tile_width, row * tile_height, tile_width, tile_height)
            image = sheet.subsurface(rect).convert_alpha()
            # Escalamos el tile individual a TILE_SIZE
            if tile_width != TILE_SIZE or tile_height != TILE_SIZE:
                image = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
            row_tiles.append(image)
        tiles.append(row_tiles)

    return tiles

def check_collision(level_data, x, y, tile_size, object_type):
    """
    Verifica colisiones en las cuatro esquinas del sprite del jugador
    Retorna True si hay colisión con una pared
    """
    # Añadimos un pequeño margen para hacer la colisión más permisiva
    margin = 10
    
    # Ajustamos las coordenadas con el margen
    check_x = x + margin
    check_y = y + margin
    check_size = tile_size - (margin * 2)

    # Convertimos las coordenadas de píxeles a coordenadas de tile
    tile_positions = [
        (check_x // tile_size, check_y // tile_size),                          # Esquina superior izquierda
        ((check_x + check_size - 1) // tile_size, check_y // tile_size),       # Esquina superior derecha
        (check_x // tile_size, (check_y + check_size - 1) // tile_size),       # Esquina inferior izquierda
        ((check_x + check_size - 1) // tile_size, (check_y + check_size - 1) // tile_size)  # Esquina inferior derecha
    ]
    
    # Verificamos cada esquina
    for tile_x, tile_y in tile_positions:
        # Verificamos que las coordenadas estén dentro de los límites del mapa
        if (0 <= tile_x < len(level_data[0]) and 
            0 <= tile_y < len(level_data) and 
            level_data[tile_y][tile_x] == object_type):
            return True
    
    return False


def wrap_text(text, font, max_width):
    """
    Divide el texto en líneas que no excedan max_width
    """
    words = text.split(' ')
    lines = []
    current_line = []
    current_width = 0

    for word in words:
        word_surface = font.render(word + ' ', True, TEXT_COLOR)
        word_width = word_surface.get_width()

        if current_width + word_width <= max_width:
            current_line.append(word)
            current_width += word_width
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_width = word_width

    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

async def main():
    pygame.init()

    game_over = False
    font_large = pygame.font.Font(None, 64)  # Fuente más grande para el mensaje final

    # Cargamos el nivel
    level_data = load_map('level.txt')
    rows = len(level_data)
    cols = len(level_data[0]) if rows > 0 else 0

    # Creamos la ventana
    screen_width = cols * TILE_SIZE
    screen_height = rows * TILE_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Mi Primer Mapa Tileado")

    # Cargamos los tilesets
    player = load_tileset('assets/Player.png', 48, 48)
    enemies = load_tileset('assets/Enemies.png', 32, 32)
    floor = load_tileset('assets/Sand.png', 32, 32)
    wall = load_tileset('assets/Wall2.png', 32, 32)
    npcs = load_tileset('assets/Npcs4.png', 96, 96)
    items = load_tileset('assets/Items.png', 32, 32)

    embeddings = init_embeddings()
    llm = init_llm()
    vectorstore = init_vectorstore(embeddings)

    # Variables del jugador
    player_pos = None
    player_direction = 'right'
    animation_frame = 0
    animation_timer = 0
    ANIMATION_SPEED = 100
    PLAYER_SPEED = 5

    last_npc_interaction_time = 0
    NPC_INTERACTION_COOLDOWN = 2000  # 2 segundos en milisegundos

    # Configuración de fuentes y chat
    font = pygame.font.Font(None, 32)
    chat_active = False
    chat_text = ""
    messages_npc1 = []
    messages_npc2 = []
    TEXT_COLOR = (255, 255, 255)
    CHAT_HEIGHT = 200
    CHAT_MARGIN = 20
    LINE_HEIGHT = 30
    INPUT_HEIGHT = 40
    SCROLLBAR_WIDTH = 20
    MAX_VISIBLE_LINES = (CHAT_HEIGHT - INPUT_HEIGHT - CHAT_MARGIN) // LINE_HEIGHT
    scroll_offset = 0
    dragging_scrollbar = False
    last_message_count = 0  # Para detectar nuevos mensajes
    should_autoscroll = True  # Nueva variable para controlar el autoscroll

    # Variables para el estado del diálogo
    npc_type = None

    npc1_book = load_random_book()

    # Diccionario de animaciones del jugador
    player_animations = {
        'right': [player[2][0], player[2][1], player[2][2]],
        'left': [player[1][0], player[1][1], player[1][2]],
        'up': [player[3][0], player[3][1], player[3][2]],
        'down': [player[0][0], player[0][1], player[0][2]],
    }

    # Mapeamos símbolos a tiles
    tile_mapping = {
        '#': wall[0][0],
        '.': floor[0][0],
        'E': enemies[0][0],
        '1': npcs[0][1],
        '2': npcs[4][1],
        'W': items[0][12]
    }

    # Encontrar la posición inicial del jugador
    for row_index, row in enumerate(level_data):
        for col_index, tile_char in enumerate(row):
            if tile_char == 'P':
                player_pos = [col_index * TILE_SIZE, row_index * TILE_SIZE]
                break

    clock = pygame.time.Clock()
    running = True
    
    while running:
        current_time = pygame.time.get_ticks()
               
        # Manejo de eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Si el juego ha terminado, solo procesar el evento de salida
            if game_over:
                continue

            # Manejo de eventos del chat
            if chat_active:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Clic izquierdo
                        mouse_pos = pygame.mouse.get_pos()
                        scrollbar_rect = pygame.Rect(
                            screen_width - SCROLLBAR_WIDTH - CHAT_MARGIN,
                            screen_height - CHAT_HEIGHT + CHAT_MARGIN,
                            SCROLLBAR_WIDTH,
                            CHAT_HEIGHT - INPUT_HEIGHT - CHAT_MARGIN
                        )
                        if scrollbar_rect.collidepoint(mouse_pos):
                            dragging_scrollbar = True
                            should_autoscroll = False  # Desactivamos autoscroll al usar scrollbar

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        dragging_scrollbar = False

                elif event.type == pygame.MOUSEMOTION:
                    if dragging_scrollbar:
                        # Calcular la nueva posición del scroll basada en la posición del ratón
                        mouse_y = event.pos[1]
                        scrollbar_top = screen_height - CHAT_HEIGHT + CHAT_MARGIN
                        scrollbar_height = CHAT_HEIGHT - INPUT_HEIGHT - CHAT_MARGIN
                        relative_y = mouse_y - scrollbar_top
                        
                        # Convertir la posición del ratón a índice de scroll
                        if len(total_lines) > MAX_VISIBLE_LINES:
                            scroll_ratio = relative_y / scrollbar_height
                            scroll_offset = int(scroll_ratio * (len(total_lines) - MAX_VISIBLE_LINES))
                            scroll_offset = max(0, min(scroll_offset, len(total_lines) - MAX_VISIBLE_LINES))

                elif event.type == pygame.MOUSEWHEEL:
                    should_autoscroll = False  # Desactivamos autoscroll al usar la rueda
                    if event.y > 0:  # Scroll arriba
                        scroll_offset = max(0, scroll_offset - 1)
                    else:  # Scroll abajo
                        if len(total_lines) > MAX_VISIBLE_LINES:
                            scroll_offset = min(scroll_offset + 1, len(total_lines) - MAX_VISIBLE_LINES)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:  # Enviar mensaje
                        if chat_text.strip():  # Si el mensaje no está vacío
                            # Añadimos el mensaje del jugador
                            if npc_type == '1':  # Si estamos hablando con NPC1
                                messages_npc1.append(f"Jugador: {chat_text}")
                                # Creamos la tarea asíncrona y esperamos su resultado
                                task = asyncio.create_task(get_npc1_response(messages_npc1, vectorstore, llm, npc1_book['name']))
                                # Cuando la tarea termine, activamos el autoscroll
                                task.add_done_callback(lambda _: setattr(sys.modules[__name__], 'should_autoscroll', True))
                            elif npc_type == '2':  # Si estamos hablando con NPC2
                                messages_npc2.append(f"Jugador: {chat_text}")
                                # Creamos la tarea asíncrona y esperamos su resultado
                                task = asyncio.create_task(get_npc2_response(messages_npc2, vectorstore, llm))
                                # Cuando la tarea termine, activamos el autoscroll
                                task.add_done_callback(lambda _: setattr(sys.modules[__name__], 'should_autoscroll', True))
                            chat_text = ""
                            should_autoscroll = True  # Activamos autoscroll al enviar mensaje
                    elif event.key == pygame.K_ESCAPE:  # Cerrar chat
                        chat_active = False
                        chat_text = ""
                        npc_type = None
                        last_npc_interaction_time = pygame.time.get_ticks()  # Actualizar el tiempo al cerrar el chat
                        continue
                    elif event.key == pygame.K_BACKSPACE:  # Borrar
                        chat_text = chat_text[:-1]
                    else:
                        # Añadir caracteres al mensaje (limitado a 50 caracteres)
                        if event.unicode.isprintable() and len(chat_text) < 200:
                            chat_text += event.unicode

        # Dar tiempo al bucle de eventos para procesar tareas asíncronas
        await asyncio.sleep(0)

        # Si el juego ha terminado, mostrar la pantalla de fin
        if game_over:
            # Fondo negro semi-transparente
            overlay = pygame.Surface((screen_width, screen_height))
            overlay.fill((0, 0, 0))
            overlay.set_alpha(180)
            screen.blit(overlay, (0, 0))
            
            # Texto de victoria
            victory_text = "¡Enhorabuena, has ayudado a Jayce a recordar el libro!"
            text_surface = font_large.render(victory_text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(screen_width/2, screen_height/2))
            screen.blit(text_surface, text_rect)
            
            pygame.display.flip()
            continue
        
        # Solo procesamos movimiento si no estamos en chat
        if not chat_active:
            # Manejo del movimiento
            keys = pygame.key.get_pressed()
            moving = False
            new_pos = player_pos.copy()

            if keys[pygame.K_LEFT]:
                new_pos[0] -= PLAYER_SPEED
                player_direction = 'left'
                moving = True
            elif keys[pygame.K_RIGHT]:
                new_pos[0] += PLAYER_SPEED
                player_direction = 'right'
                moving = True
            elif keys[pygame.K_UP]:
                new_pos[1] -= PLAYER_SPEED
                player_direction = 'up'
                moving = True
            elif keys[pygame.K_DOWN]:
                new_pos[1] += PLAYER_SPEED
                player_direction = 'down'
                moving = True

            # Verificar colisiones y actualizar posición
            new_pos[0] = max(0, min(new_pos[0], screen_width - TILE_SIZE))
            new_pos[1] = max(0, min(new_pos[1], screen_height - TILE_SIZE))

            if not check_collision(level_data, new_pos[0], new_pos[1], TILE_SIZE, '#'):
                player_pos = new_pos

                # Check collision with the item
                if check_collision(level_data, player_pos[0], player_pos[1], TILE_SIZE, 'W'):
                    game_over = True

                # Verificar colisión con NPCs solo si ha pasado suficiente tiempo
                elif current_time - last_npc_interaction_time >= NPC_INTERACTION_COOLDOWN:
                    npc1_collision = check_collision(level_data, player_pos[0], player_pos[1], TILE_SIZE, '1')
                    npc2_collision = check_collision(level_data, player_pos[0], player_pos[1], TILE_SIZE, '2')
                    if npc1_collision:
                        chat_active = True
                        npc_type = '1'  # Marcamos que estamos hablando con NPC1
                        asyncio.create_task(get_npc1_response(messages_npc1, vectorstore, llm, npc1_book['name']))
                    elif npc2_collision:
                        chat_active = True
                        npc_type = '2'  # Marcamos que estamos hablando con NPC2
                        asyncio.create_task(get_npc2_response(messages_npc2, vectorstore, llm))

                

            # Actualizar animación
            if moving:
                if current_time - animation_timer > ANIMATION_SPEED:
                    animation_frame = (animation_frame + 1) % len(player_animations[player_direction])
                    animation_timer = current_time
            else:
                animation_frame = 1

        # Pintamos el fondo
        screen.fill((135, 206, 235))

        # Dibujamos todo el mapa y entidades en un solo bucle
        for row_index, row in enumerate(level_data):
            for col_index, tile_char in enumerate(row):
                x = col_index * TILE_SIZE
                y = row_index * TILE_SIZE
                
                # Siempre dibujamos el suelo
                screen.blit(floor[0][0], (x, y))
                
                # Dibujamos paredes y enemigos, pero no el cofre
                if tile_char in tile_mapping:
                    if tile_char != 'W' or BOOK_REMEMBERED:
                        screen.blit(tile_mapping[tile_char], (x, y))

        # Dibujamos al jugador
        current_frame = player_animations[player_direction][animation_frame]
        screen.blit(current_frame, player_pos)

        # Dibujar interfaz de chat si está activo
        if chat_active:
            # Fondo del chat
            chat_surface = pygame.Surface((screen_width, CHAT_HEIGHT))
            chat_surface.fill((0, 0, 0))
            chat_surface.set_alpha(180)
            screen.blit(chat_surface, (0, screen_height - CHAT_HEIGHT))

            # Calcular todas las líneas
            total_lines = []
            if npc_type == '1':
                # Detectar nuevos mensajes para autoscroll
                if len(messages_npc1) > last_message_count:
                    scroll_offset = 0  # Reset al fondo
                    last_message_count = len(messages_npc1)

                for msg in messages_npc1:
                    wrapped_lines = wrap_text(msg, font, screen_width - CHAT_MARGIN * 3 - SCROLLBAR_WIDTH)
                    total_lines.extend(wrapped_lines)
            elif npc_type == '2':
                # Detectar nuevos mensajes para autoscroll
                if len(messages_npc2) > last_message_count:
                    scroll_offset = 0  # Reset al fondo
                    last_message_count = len(messages_npc2)

                for msg in messages_npc2:
                    wrapped_lines = wrap_text(msg, font, screen_width - CHAT_MARGIN * 3 - SCROLLBAR_WIDTH)
                    total_lines.extend(wrapped_lines)

            # Autoscroll si está activado y hay suficientes líneas
            if should_autoscroll and len(total_lines) > MAX_VISIBLE_LINES:
                scroll_offset = len(total_lines) - MAX_VISIBLE_LINES
            
            # Asegurar que el scroll_offset es válido
            if len(total_lines) > MAX_VISIBLE_LINES:
                scroll_offset = max(0, min(scroll_offset, len(total_lines) - MAX_VISIBLE_LINES))
            else:
                scroll_offset = 0
            
            # Área de mensajes (con margen para el scrollbar)
            messages_rect = pygame.Rect(
                CHAT_MARGIN, 
                screen_height - CHAT_HEIGHT + CHAT_MARGIN,
                screen_width - CHAT_MARGIN * 2 - SCROLLBAR_WIDTH,
                CHAT_HEIGHT - INPUT_HEIGHT - CHAT_MARGIN
            )

            # Dibujar los mensajes visibles
            y_offset = messages_rect.top
            visible_lines = total_lines[scroll_offset:scroll_offset + MAX_VISIBLE_LINES]
            for line in visible_lines:
                text_surface = font.render(line, True, TEXT_COLOR)
                screen.blit(text_surface, (messages_rect.left, y_offset))
                y_offset += LINE_HEIGHT

            # Dibujar scrollbar si hay más líneas que las visibles
            if len(total_lines) > MAX_VISIBLE_LINES:
                scrollbar_height = messages_rect.height
                scroll_thumb_height = max(20, scrollbar_height * (MAX_VISIBLE_LINES / len(total_lines)))
                scroll_thumb_pos = scrollbar_height * (scroll_offset / (len(total_lines) - MAX_VISIBLE_LINES))
                
                # Fondo del scrollbar
                scrollbar_rect = pygame.Rect(
                    screen_width - SCROLLBAR_WIDTH - CHAT_MARGIN,
                    messages_rect.top,
                    SCROLLBAR_WIDTH,
                    scrollbar_height
                )
                pygame.draw.rect(screen, (50, 50, 50), scrollbar_rect)
                
                # Thumb del scrollbar
                thumb_rect = pygame.Rect(
                    screen_width - SCROLLBAR_WIDTH - CHAT_MARGIN,
                    messages_rect.top + scroll_thumb_pos,
                    SCROLLBAR_WIDTH,
                    scroll_thumb_height
                )
                pygame.draw.rect(screen, (150, 150, 150), thumb_rect)

            # Área de input (con fondo más oscuro para distinguirla)
            input_rect = pygame.Rect(
                CHAT_MARGIN,
                screen_height - INPUT_HEIGHT - CHAT_MARGIN,
                screen_width - CHAT_MARGIN * 2,
                INPUT_HEIGHT
            )
            pygame.draw.rect(screen, (30, 30, 30), input_rect)
            
            # Texto del input
            input_text = font.render(f"> {chat_text}", True, TEXT_COLOR)
            screen.blit(input_text, (
                input_rect.left + 5,
                input_rect.top + (input_rect.height - font.get_height()) // 2
            ))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    asyncio.run(main())
