import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from PIL import Image

class PersonDetector:
    def __init__(self):
        # CUDA 초기화
        self.cfx = cuda.Device(0).make_context()
        
        # TensorRT 로거 생성
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # TensorRT 엔진 로드
        engine_path = "/opt/nvidia/deepstream/deepstream/samples/models/PeopleNet/resnet34_peoplenet.onnx_b1_gpu0_int8.engine"
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()

        # 입출력 바인딩 준비
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        # 입출력 메모리 할당
        for binding in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(binding)
            shape = self.engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            # 입력 텐서는 일반적으로 첫 번째 텐서입니다
            if binding == 0:
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

        # Add input shape information
        self.input_shape = self.engine.get_tensor_shape(self.engine.get_tensor_name(0))
        self.context.set_input_shape(self.engine.get_tensor_name(0), self.input_shape)

    def preprocess_image(self, frame):
        # OpenCV BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 크기 조정 및 정규화
        image = cv2.resize(image, (960, 544))
        image = image.astype(np.float32) / 255.0
        
        # CHW 형식으로 변환
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def detect(self, frame):
        # 이미지 전처리
        input_image = self.preprocess_image(frame)
        
        # 입력 데이터 복사
        np.copyto(self.inputs[0]["host"], input_image.ravel())
        
        # GPU로 데이터 전송
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
        
        # Set tensor addresses before execution
        for binding_idx, binding in enumerate(self.bindings):
            if binding_idx == 0:
                self.context.set_tensor_address(self.engine.get_tensor_name(binding_idx), self.inputs[0]["device"])
            else:
                self.context.set_tensor_address(self.engine.get_tensor_name(binding_idx), self.outputs[binding_idx-1]["device"])
        
        # 추론 실행
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 결과 가져오기
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        
        # 동기화
        self.stream.synchronize()
        
        # 결과 처리
        outputs = []
        for out in self.outputs:
            outputs.append(out["host"])
        
        # 바운딩 박스 처리 수정
        detection_out = outputs[0]
        scores = outputs[1]
        
        # Print shapes for debugging
        print(f"Detection shape: {detection_out.shape}")
        print(f"Scores shape: {scores.shape}")
        
        # PeopleNet의 출력 형식에 맞게 reshape
        # scores는 (102, 240) 형태로 reshape (24480 = 102 * 240)
        scores = scores.reshape(102, 240)
        # boxes도 같은 grid 크기로 reshape (6120 = 102 * 15 * 4)
        boxes = detection_out.reshape(102, 15, 4)
        
        # 신뢰도 임계값 적용 (grid 별로)
        confidence_threshold = 0.5
        selected_boxes = []
        
        for i in range(scores.shape[0]):
            grid_scores = scores[i]
            grid_boxes = boxes[i]
            mask = grid_scores > confidence_threshold
            selected_boxes.append(grid_boxes[mask[:15]])  # 첫 15개의 confidence score만 사용
        
        # 선택된 박스들을 하나의 배열로 합치기
        if selected_boxes:
            boxes = np.vstack(selected_boxes)
        else:
            boxes = np.array([])
        
        if len(boxes) > 0:
            # 박스 좌표를 원본 이미지 크기에 맞게 조정
            h, w = frame.shape[:2]
            boxes[:, [0, 2]] *= w / 960
            boxes[:, [1, 3]] *= h / 544
        
        return boxes.astype(np.int32)

    def __del__(self):
        try:
            self.cfx.pop()
        except:
            pass

def main():
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    detector = PersonDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # 사람 검출
            boxes = detector.detect(frame)
            
            # 결과 시각화
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # ROI 내 사람 수 계산
            roi = np.array([[789, 672], [1084, 900], [851, 773], [1203, 732]])
            person_count = 0
            for box in boxes:
                center_x = (box[0] + box[2]) // 2
                center_y = (box[1] + box[3]) // 2
                point = (int(center_x), int(center_y))
                if cv2.pointPolygonTest(roi, point, False) >= 0:
                    person_count += 1

            # ROI 그리기
            cv2.polylines(frame, [roi], True, (255, 0, 0), 2)
            cv2.putText(frame, f"People in ROI: {person_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error during detection: {e}")

        # 화면 표시
        cv2.imshow("Person Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()