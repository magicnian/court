import os

img_dir = 'E:\\document\\ocr\\court\\zhixing\\dst'


def transfer():
    old_imgs = os.listdir(img_dir)
    for i, old_img in enumerate(old_imgs):
        old_name = old_img.split('.')[0]
        old_type = old_img.split('.')[1]

        new_name = old_name.lower()
        new_type = old_type

        src = os.path.join(img_dir, old_img)
        dst = os.path.join(img_dir, new_name + '.' + new_type)
        os.rename(src, dst)
        print(i)
    print('done')


def remove_useless():
    old_imgs = os.listdir(img_dir)
    for i, old_img in enumerate(old_imgs):
        if len(old_img.split('-')[0])!=4:
            os.remove(os.path.join(img_dir, old_img))
            print('done')

def delete_error():
    captcha_word = "0123456789abcdefghijklmnopqrstuvwxyz"
    old_imgs = os.listdir(img_dir)
    for i, old_img in enumerate(old_imgs):
        old_name = old_img.split('-')[0]
        for ch in old_name:
            if ch not in captcha_word:
                print(old_name)
                os.remove(os.path.join(img_dir, old_img))
                continue

if __name__ == '__main__':
    transfer()
    # remove_useless()
    # delete_error()