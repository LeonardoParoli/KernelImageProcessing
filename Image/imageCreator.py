def create_ppm_image(width, height, center_x, center_y, num_circles):
    image = [['255 255 255'] * width for _ in range(height)]
    circle_radius = min(width, height) // (2 * num_circles)
    for i in range(num_circles):
        radius = circle_radius * (i + 1)
        draw_circle(image, center_x, center_y, radius)
    save_ppm_image(image, width, height, 'image.ppm')


def draw_circle(image, center_x, center_y, radius):
    for y in range(len(image)):
        for x in range(len(image[y])):
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            if abs(distance - radius) <= 3:
                image[y][x] = '0 0 0'
def save_ppm_image(image, width, height, filename):
    with open(filename, 'w') as file:
        file.write('P3\n')
        file.write(f'{width} {height}\n')
        file.write('255\n')
        for row in image:
            file.write(' '.join(row) + '\n')

image_width = 800
image_height = 800
circle_center_x = image_width // 2
circle_center_y = image_height // 2
num_of_circles = 25
create_ppm_image(image_width, image_height, circle_center_x, circle_center_y, num_of_circles)