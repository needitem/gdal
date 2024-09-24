import os
from osgeo import gdal
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import logging


class Preprocessing:
    def __init__(self, image_path, histogram_enhancement=True, auto_save=False):
        """
        이미지 전처리 클래스 초기화.

        :param image_path: 처리할 이미지 파일 경로
        :param histogram_enhancement: 히스토그램 평활화 적용 여부
        :param auto_save: 처리 후 자동 저장 여부
        """
        self.image_path = image_path
        self.auto_save = auto_save
        self._setup_logging()

        try:
            self.image = gdal.Open(image_path)
            if self.image is None:
                raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")
            self.rb = self.image.GetRasterBand(1)
            self.image_array = self.rb.ReadAsArray()

            if self.image_array.dtype != np.uint8:
                self.image_array = cv2.normalize(
                    self.image_array, None, 0, 255, cv2.NORM_MINMAX
                )
                self.image_array = self.image_array.astype(np.uint8)
                logging.info("이미지 정규화 완료.")

            if histogram_enhancement:
                self.image_array = cv2.equalizeHist(self.image_array)
                logging.info("히스토그램 평활화 완료.")
        except Exception as e:
            logging.error(f"초기화 중 오류 발생: {e}")
            sys.exit(1)

    def _setup_logging(self):
        """로깅 설정."""
        logging.basicConfig(
            filename="preprocessing.log",
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s",
        )

    def show_image(self):
        """이미지를 화면에 표시합니다."""
        try:
            plt.imshow(self.image_array, cmap="gray")
            plt.title("Processed Image")
            plt.axis("off")
            plt.show()
            logging.info("이미지 표시 완료.")
        except Exception as e:
            logging.error(f"이미지 표시 중 오류 발생: {e}")

    def save_image(self, save_path=None):
        """
        이미지를 저장합니다.

        :param save_path: 저장할 경로. 지정하지 않으면 원본 경로에 덮어씁니다.
        """
        try:
            path = save_path if save_path else self.image_path
            cv2.imwrite(path, self.image_array)
            logging.info(f"이미지 저장 완료: {path}")
        except Exception as e:
            logging.error(f"이미지 저장 중 오류 발생: {e}")

    def get_image(self):
        """이미지 배열을 반환합니다."""
        return self.image_array

    def get_image_array(self):
        """이미지 배열을 반환합니다."""
        return self.image_array

    def get_image_path(self):
        """이미지 경로를 반환합니다."""
        return self.image_path

    def get_image_dtype(self):
        """이미지 데이터 타입을 반환합니다."""
        return self.image_array.dtype

    def rotate_image(self, angle, save_after=False, save_path=None):
        """
        이미지를 회전시킵니다.

        :param angle: 회전 각도
        :param save_after: 회전 후 저장 여부
        :param save_path: 저장할 경로
        """
        try:
            rows, cols = self.image_array.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            self.image_array = cv2.warpAffine(self.image_array, M, (cols, rows))
            logging.info(f"이미지 {angle}도 회전 완료.")

            if save_after:
                self.save_image(save_path)
        except Exception as e:
            logging.error(f"이미지 회전 중 오류 발생: {e}")

    def resize_image(
        self,
        width,
        height,
        interpolation=cv2.INTER_LINEAR,
    ):
        """
        이미지 크기를 조정합니다.

        :param width: 새 너비
        :param height: 새 높이
        :param interpolation: 보간 방법
        """
        try:
            self.image_array = cv2.resize(
                self.image_array, (width, height), interpolation=interpolation
            )
            logging.info(f"이미지 크기 조정 완료: {width}x{height}")

        except Exception as e:
            logging.error(f"이미지 크기 조정 중 오류 발생: {e}")

    def crop_image(self, x, y, width, height, save_after=False, save_path=None):
        """
        이미지를 잘라냅니다.

        :param x: 시작 x 좌표
        :param y: 시작 y 좌표
        :param width: 잘라낼 너비
        :param height: 잘라낼 높이
        :param save_after: 자른 후 저장 여부
        :param save_path: 저장할 경로
        """
        try:
            self.image_array = self.image_array[y : y + height, x : x + width]
            logging.info(f"이미지 크롭 완료: ({x}, {y}, {width}, {height})")

            if save_after:
                self.save_image(save_path)
        except Exception as e:
            logging.error(f"이미지 크롭 중 오류 발생: {e}")

    def adjust_resolution(
        self,
        width,
        height,
        interpolation=cv2.INTER_LINEAR,
    ):
        """
        이미지 해상도를 조정합니다.

        :param width: 새 너비
        :param height: 새 높이
        :param interpolation: 보간 방법
        """
        self.resize_image(width, height, interpolation)

    def add_gaussian_noise(self, mean=0, var=0.01, save_after=False, save_path=None):
        """
        이미지에 가우시안 노이즈를 추가합니다.

        :param mean: 노이즈 평균
        :param var: 노이즈 분산
        :param save_after: 노이즈 추가 후 저장 여부
        :param save_path: 저장할 경로
        """
        try:
            sigma = var**0.5
            gaussian = np.random.normal(mean, sigma, self.image_array.shape)
            noisy_image = self.image_array + gaussian
            self.image_array = np.clip(noisy_image, 0, 255).astype(np.uint8)
            logging.info("가우시안 노이즈 추가 완료.")

            if save_after:
                self.save_image(save_path)
        except Exception as e:
            logging.error(f"가우시안 노이즈 추가 중 오류 발생: {e}")

    def save_processed_image(self, suffix="_processed"):
        """
        처리된 이미지를 원본 파일명에 접미사를 추가하여 저장합니다.

        :param suffix: 파일명에 추가할 접미사
        """
        try:
            base, ext = os.path.splitext(self.image_path)
            new_path = f"{base}{suffix}{ext}"
            self.save_image(new_path)
            logging.info(f"처리된 이미지 저장 완료: {new_path}")
        except Exception as e:
            logging.error(f"처리된 이미지 저장 중 오류 발생: {e}")


if __name__ == "__main__":
    preprocessing = Preprocessing(
        ""
    )
    # preprocessing.rotate_image(60)
    # preprocessing.crop_image(1000, 1000, 1000, 1000)  # x, y, width, height
    # preprocessing.adjust_resolution(500, 500)
    preprocessing.show_image()
