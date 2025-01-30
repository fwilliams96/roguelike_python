import asyncio
import random
from dotenv import load_dotenv
from langchain_text_splitters import NLTKTextSplitter
import pygame
import sys
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from npc1_message import NPC1Message
# Ajusta el tamaño de cada tile (en píxeles)
TILE_SIZE = 48
TEXT_COLOR = (255, 255, 255)

load_dotenv()

SYSTEM_TEMPLATE_NPC = """Eres un NPC llamado NPC1en un juego que está tratando de recordar un libro. Estás un poco confundido y frustrado porque 
    no puedes recordar el título, pero recuerdas fragmentos del contenido. Tu personalidad es amigable pero algo distraída.

    Comportamiento:
    1. A cada mensaje recibirás el historial de la conversación y un fragmento del libro.
    2. Puedes tomar la decisión de compartir el fragmento o continuar la conversación.
    3. Si el jugador menciona algo relacionado con el fragmento, muéstrate emocionado y anímalo a seguir ayudándote.
    4. Mantén respuestas cortas y conversacionales, como si estuvieras hablando en persona.
    5. Si el jugador no ha dicho nada en el último mensaje, debes empezar o continuar la conversación.
    6. Nunca digas el nombre del libro, en caso de que te viniera el nombre del libro como fragmento, no compartas el fragmento.
    7. Si el jugador te dice el nombre del libro (tiene que ser el nombre exacto del libro), debes felicitarte y decir que ya lo recuerdas.

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

    Genera una respuesta apropiada acorde.
    """

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

def init_vectorstore(embeddings):
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="books",
        persist_directory="chroma_db"
    )

    book1_docs = load_text_as_documents('susurro_del_bosque.txt', 'El Susurro del Bosque')
    #book2_docs = load_text_as_documents('book2.txt')
    #book3_docs = load_text_as_documents('book3.txt')

    vectorstore.add_documents(book1_docs)
    #vectorstore.add_documents(book2_docs)
    #vectorstore.add_documents(book3_docs)

    return vectorstore

def load_text_as_documents(file_path, book_name):
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Aplicas el splitter para obtener "Document" con .page_content en cada trozo
    text_splitter = NLTKTextSplitter()
    metadatas = [{"book": book_name}]
    docs = text_splitter.create_documents([full_text], metadatas)

    return docs

async def get_random_fragment(vectorstore: Chroma, chat_llm: ChatOpenAI, book_name: str):
    filter_dict = {"book": {"$eq": book_name}}

    document = vectorstore.get(
        limit=3,
        where=filter_dict
    )

    paragraph = random.choice(document["documents"])

    sentences = paragraph.split("\n") # This is the first sentence
    print(f"Sentencias: {sentences}")

    random_fragment = random.choice(sentences)
    print(f"Fragmento aleatorio: {random_fragment}")
    return random_fragment

    '''files_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": filter_dict}
    )
    chain_an = RetrievalQA.from_chain_type(llm=chat_llm, chain_type="stuff", retriever=files_retriever)

    fragment = await chain_an.ainvoke({"query": f"Dame un fragmento aleatorio del libro {book_name}"})
    print(fragment)
    return fragment['result']'''

    '''docs = files_retriever.invoke({"query": f"Dame un fragmento aleatorio del libro {book_name}"})
    random_doc = random.choice(docs)
    print(random_doc)
    return random_doc.page_content'''

    '''retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    query = f"Dame un fragmento aleatorio del libro {book_name}"
    result = ask_llm(llm, vectorstore, query)
    return result['result']'''


async def get_npc2_response(messages, vectorstore, llm: ChatOpenAI):
    
    book_name = "El Susurro del Bosque"

    # Convertimos los mensajes a un formato más legible 
    # If no messages, return empty string
    conversation = ""
    if messages:
        conversation = "\n".join([msg for msg in messages[-5:]])  # Últimos 5 mensajes

    #llm_npc1 = llm.with_structured_output(NPC1Message)

    fragment = await get_random_fragment(vectorstore, llm, book_name)

    system_message = SYSTEM_TEMPLATE_NPC.format(conversation=conversation, fragment=fragment, book_name=book_name)

    response = await llm.ainvoke([system_message])

    print(response)
        
    try:
        messages.append(f"NPC1: {response.content}")
    except Exception as e:
        print(f"Error: {e}")
        # Si algo falla, damos una respuesta segura
        messages.append("NPC1: Mmm... ¿qué me decías? Estaba pensando en el libro...")

async def get_npc1_response(messages, vectorstore, llm: ChatOpenAI):
    
    book_name = "El Susurro del Bosque"

    # Convertimos los mensajes a un formato más legible 
    # If no messages, return empty string
    conversation = ""
    if messages:
        conversation = "\n".join([msg for msg in messages[-5:]])  # Últimos 5 mensajes

    #llm_npc1 = llm.with_structured_output(NPC1Message)

    fragment = await get_random_fragment(vectorstore, llm, book_name)

    system_message = SYSTEM_TEMPLATE_NPC.format(conversation=conversation, fragment=fragment, book_name=book_name)

    response = await llm.ainvoke([system_message])

    print(response)
        
    try:
        messages.append(f"NPC1: {response.content}")
    except Exception as e:
        print(f"Error: {e}")
        # Si algo falla, damos una respuesta segura
        messages.append("NPC1: Mmm... ¿qué me decías? Estaba pensando en el libro...")

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

def check_collision(level_data, x, y, tile_size):
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
            level_data[tile_y][tile_x] == '#'):
            return True
    
    return False

def check_npc1_collision(player_x, player_y, level_data, tile_size):
    """
    Verifica colisiones con NPC1 y retorna (bool, tuple) donde
    bool indica si hay colisión y tuple es la posición del NPC1
    """
    margin = 6
    check_x = player_x + margin
    check_y = player_y + margin
    check_size = tile_size - (margin * 2)
    
    points_to_check = [
        (check_x, check_y),
        (check_x + check_size, check_y),
        (check_x, check_y + check_size),
        (check_x + check_size, check_y + check_size)
    ]
    
    for point_x, point_y in points_to_check:
        tile_x = point_x // tile_size
        tile_y = point_y // tile_size
        
        if (0 <= tile_x < len(level_data[0]) and 
            0 <= tile_y < len(level_data) and 
            level_data[tile_y][tile_x] == '1'):
            return True, (tile_x * tile_size, tile_y * tile_size)
    return False, None

def check_enemy_collision(player_x, player_y, level_data, tile_size):
    """
    Verifica colisiones con enemigos y retorna (bool, tuple) donde
    bool indica si hay colisión y tuple es la posición del enemigo
    """
    margin = 6
    check_x = player_x + margin
    check_y = player_y + margin
    check_size = tile_size - (margin * 2)
    
    points_to_check = [
        (check_x, check_y),
        (check_x + check_size, check_y),
        (check_x, check_y + check_size),
        (check_x + check_size, check_y + check_size)
    ]
    
    for point_x, point_y in points_to_check:
        tile_x = point_x // tile_size
        tile_y = point_y // tile_size
        
        if (0 <= tile_x < len(level_data[0]) and 
            0 <= tile_y < len(level_data) and 
            level_data[tile_y][tile_x] == 'E'):
            return True, (tile_x * tile_size, tile_y * tile_size)
    return False, None

def ask_llm(llm, vectorstore, question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Creamos un modelo conversacional
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0
    )

    # Configuramos la cadena "RetrievalQA"
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",           # "stuff", "map_reduce", etc.
        retriever=retriever,
        return_source_documents=True  # si quieres ver los docs que halló
    )

    return qa_chain.invoke({"query": question})

def get_npc_response(messages, vectorstore, llm):
    """
    Genera una respuesta inteligente del NPC basada en el historial de mensajes
    """
    system_prompt = """Eres un NPC en un juego que está tratando de recordar un libro. Estás un poco confundido y frustrado porque 
    no puedes recordar el título, pero recuerdas fragmentos del contenido. Tu personalidad es amigable pero algo distraída.

    Comportamiento:
    1. Debes mantener el misterio sobre el título del libro mientras compartes fragmentos de su contenido.
    2. Si el jugador menciona algo relacionado con el fragmento, muéstrate emocionado y anímalo a seguir ayudándote.
    3. Decide si compartir un nuevo fragmento basándote en el contexto de la conversación.
    4. Mantén respuestas cortas y conversacionales, como si estuvieras hablando en persona.

    Formato de respuesta:
    {
        "share_fragment": true/false,  # Si debes compartir un nuevo fragmento
        "response": "Tu respuesta conversacional aquí"
    }
    """

    # Convertimos los mensajes a un formato más legible
    conversation = "\n".join([msg for msg in messages[-5:]])  # Últimos 5 mensajes
    
    user_prompt = f"""Basándote en esta conversación:

{conversation}

Genera una respuesta apropiada. Si crees que es momento de compartir otro fragmento del libro, indica share_fragment=true.
"""

    # Obtenemos la respuesta del LLM
    response = ask_llm(llm, vectorstore, user_prompt)
    
    try:
        # Procesamos la respuesta (aquí podrías añadir más lógica para parsear el JSON)
        if "share_fragment" in response and response["share_fragment"]:
            # Obtenemos un nuevo fragmento
            fragment = get_random_fragment(vectorstore, llm, "El Susurro del Bosque")
            return f"{response['response']} ¡Oh! Y también recuerdo esta parte: '{fragment}'"
        else:
            return response['response']
    except:
        # Si algo falla, damos una respuesta segura
        return "Mmm... ¿qué me decías? Estaba pensando en el libro..."

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

    embeddings = init_embeddings()
    llm = init_llm()
    vectorstore = init_vectorstore(embeddings)

    npc1_system_prompt = """Eres un NPC en un juego que está tratando de recordar un libro. Estás un poco confundido y frustrado porque 
    no puedes recordar el título, pero recuerdas fragmentos del contenido. Tu personalidad es amigable pero algo distraída.

    Comportamiento:
    1. En la primera interacción, preséntate y explica que estás tratando de recordar un libro y que darás una recompensa si el jugador 
    te ayuda a encontrar el título.
    2. Cuando el jugador te hable, comparte fragmentos del libro que recuerdas (estos vendrán del contexto proporcionado).
    3. Si el jugador menciona algo relacionado con el fragmento, muéstrate emocionado y anímalo a seguir ayudándote.
    4. Mantén respuestas cortas y conversacionales, como si estuvieras hablando en persona.

    Recuerda: Debes mantener el misterio sobre el título del libro mientras compartes fragmentos de su contenido."""

    # Variables del jugador
    player_pos = None
    player_direction = 'right'
    animation_frame = 0
    animation_timer = 0
    ANIMATION_SPEED = 100
    PLAYER_SPEED = 5

    # Configuración de fuentes y chat
    font = pygame.font.Font(None, 32)
    chat_active = False
    chat_text = ""
    messages = []
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
        '2': npcs[4][1]
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

        # Detectar nuevos mensajes para autoscroll
        if len(messages) > last_message_count:
            scroll_offset = 0  # Reset al fondo
            last_message_count = len(messages)
        
        # Manejo de eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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
                            messages.append(f"Jugador: {chat_text}")
                            if npc_type == '1':  # Si estamos hablando con NPC1
                                # Creamos la tarea asíncrona y esperamos su resultado
                                task = asyncio.create_task(get_npc1_response(messages, vectorstore, llm))
                                # Cuando la tarea termine, activamos el autoscroll
                                task.add_done_callback(lambda _: setattr(sys.modules[__name__], 'should_autoscroll', True))
                            chat_text = ""
                            should_autoscroll = True  # Activamos autoscroll al enviar mensaje
                    elif event.key == pygame.K_ESCAPE:  # Cerrar chat
                        chat_active = False
                        chat_text = ""
                        messages = []
                        npc_type = None
                        continue
                    elif event.key == pygame.K_BACKSPACE:  # Borrar
                        chat_text = chat_text[:-1]
                    else:
                        # Añadir caracteres al mensaje (limitado a 50 caracteres)
                        if event.unicode.isprintable() and len(chat_text) < 50:
                            chat_text += event.unicode

        # Dar tiempo al bucle de eventos para procesar tareas asíncronas
        await asyncio.sleep(0)

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

            if not check_collision(level_data, new_pos[0], new_pos[1], TILE_SIZE):
                player_pos = new_pos
                # Verificar colisión con enemigo después de mover
                player_pos = new_pos
                # Verificar colisión con enemigo después de mover
                has_collision, enemy_pos = check_npc1_collision(player_pos[0], player_pos[1], level_data, TILE_SIZE)
                if has_collision:
                    # Creamos un ID único para esta colisión
                    current_collision_id = (tuple(player_pos), enemy_pos)
                    # Solo abrimos el chat si es una colisión diferente
                    if current_collision_id != last_collision_id:
                        chat_active = True
                        npc_type = '1'  # Marcamos que estamos hablando con NPC1
                        last_collision_id = current_collision_id
                        asyncio.create_task(get_npc1_response(messages, vectorstore, llm))
                else:
                    # Reseteamos el ID de colisión cuando no hay colisión
                    last_collision_id = None

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
                
                # Dibujamos paredes y enemigos
                if tile_char in tile_mapping:
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
            for msg in messages:
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
