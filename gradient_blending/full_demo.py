from seamlessCloningPoisson import *
import createMask
# Demo to show gradient blending. Make sure both the source and target images are JPEGs!
def main():
    output_file = 'eevee'
    input_src = 'demo/133.jpg'
    input_target = 'demo/table.jpg'
    input_mask = 'demo/133_mask.png'

    offsetX=130
    offsetY=160

    src = imread(input_src)
    target = imread(input_target)
    createMask.main(input_src)
    mask = imread(input_mask)

    res = seamlessCloningPoisson(src, target, mask, offsetX, offsetY)
    fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    plt.imshow(res)
    plt.axis('off')
    plt.savefig('eevee',bbox_inches='tight',pad_inches = 0)
    print('Image saved to ' + output_file + '.png')
if __name__== "__main__":
    main()
 
