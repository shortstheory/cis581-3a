from seamlessCloningPoisson import *

# Demo to show gradient blending. Make sure both the source and target images are JPEGs!
def main():
    output_file = 'eevee'
    mask = imread('images/133_mask.png')
    src = imread('images/133.jpg')
    target = imread('images/table.jpg')
    offsetX=130
    offsetY=160

    res = seamlessCloningPoisson(src, target, mask, offsetX, offsetY)
    fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    plt.imshow(res)
    plt.axis('off')
    plt.savefig('eevee',bbox_inches='tight',pad_inches = 0)
    print('Image saved to ' + output_file + '.png')
if __name__== "__main__":
    main()
