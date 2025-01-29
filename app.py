import pygame
import sys

# Ajusta el tamaño de cada tile (en píxeles)
TILE_SIZE = 48

def load_map(filename):
    with open(filename, 'r') as f:
        lines = [line.strip('\n') for line in f]
    return lines


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

def main():
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
    messages = []  # Lista para almacenar los mensajes
    TEXT_COLOR = (255, 255, 255)
    CHAT_BG_COLOR = (0, 0, 0, 180)  # Negro semi-transparente
    last_collision_id = None  # Tupla (player_pos, enemy_pos)

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

            # Manejo de eventos del chat
            if chat_active:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:  # Enviar mensaje
                        if chat_text.strip():  # Si el mensaje no está vacío
                            messages.append(chat_text)
                            chat_text = ""
                    elif event.key == pygame.K_ESCAPE:  # Cerrar chat
                        chat_active = False
                        chat_text = ""
                        messages = []
                        continue  # Aseguramos que se procese el escape
                    elif event.key == pygame.K_BACKSPACE:  # Borrar
                        chat_text = chat_text[:-1]
                    else:
                        # Añadir caracteres al mensaje (limitado a 50 caracteres)
                        if len(chat_text) < 50:
                            chat_text += event.unicode

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
                has_collision, enemy_pos = check_enemy_collision(player_pos[0], player_pos[1], level_data, TILE_SIZE)
                if has_collision:
                    # Creamos un ID único para esta colisión
                    current_collision_id = (tuple(player_pos), enemy_pos)
                    # Solo abrimos el chat si es una colisión diferente
                    if current_collision_id != last_collision_id:
                        chat_active = True
                        last_collision_id = current_collision_id
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
            chat_surface = pygame.Surface((screen_width, 200))
            chat_surface.fill((0, 0, 0))
            chat_surface.set_alpha(180)
            screen.blit(chat_surface, (0, screen_height - 200))

            # Mostrar mensajes anteriores
            y_offset = screen_height - 190
            for msg in messages[-5:]:  # Mostrar últimos 5 mensajes
                text_surface = font.render(msg, True, TEXT_COLOR)
                screen.blit(text_surface, (10, y_offset))
                y_offset += 30

            # Mostrar texto actual
            input_text = font.render(f"> {chat_text}", True, TEXT_COLOR)
            screen.blit(input_text, (10, screen_height - 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
