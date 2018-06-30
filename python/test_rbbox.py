import unittest
import rbbox

class TestRbboxMethods(unittest.TestCase):
    
    def test_get_result_filename(self):
        image_path = "path/to/image/img.png"
        result_path = rbbox.get_result_filename(image_path, suffix="-result-resnet50")
        self.assertEqual(result_path, "path/to/image/img-result-resnet50.txt")

    def test_save_result(self):
        image_path = "img.png"
        labels = [0, 1]
        boxes = [[0, 1, 2, 3], [4, 5, 6, 7]]
        rboxes = [[0, 0, 1, 1, 90], [2, 2, 3, 3, 45]]
        rbbox.save_result(image_path, labels, boxes, rboxes)

if __name__ == '__main__':
    unittest.main()