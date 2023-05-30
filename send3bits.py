import cv2
import numpy

pixels_per_row = 1900
bytes_per_row = pixels_per_row * 4
start_code = numpy.array([255,127,85,127,255], dtype=numpy.uint8)

total_sequences = 12
file_name = "file"

for file_index in range(total_sequences):
    with open(file_name + ".7z." + str(file_index + 1).zfill(3), mode="rb") as zip_file:
        bytes_contents = zip_file.read()

    byte_array = list(bytes_contents)
    data = numpy.array(byte_array, dtype=numpy.uint8)
    bytes_in_frame = data.size

    pixels_per_row_array = numpy.array([pixels_per_row % 256,pixels_per_row // 256], dtype=numpy.uint8) # 2 bytes
    total_sequences_array = numpy.array([total_sequences % 256,total_sequences // 256], dtype=numpy.uint8) # 2 bytes
    message_sequence_array = numpy.array([file_index % 256,file_index // 256], dtype=numpy.uint8) # 2 btytes    
    bytes_in_frame_array = numpy.array([bytes_in_frame % 256,(bytes_in_frame // 256) % 256, bytes_in_frame // 65536], dtype=numpy.uint8) # 3 bytes
    
    packed_data = numpy.concatenate((start_code, pixels_per_row_array, total_sequences_array, message_sequence_array, bytes_in_frame_array, data), axis=0)
    unpacked_data = numpy.unpackbits(packed_data, axis=0) * 255

    bits_per_row = 3 * pixels_per_row
    pad_last_row = numpy.zeros((bits_per_row - ((len(unpacked_data)) % bits_per_row)) % bits_per_row, dtype=numpy.uint8)
    unpacked_data = numpy.concatenate((unpacked_data, pad_last_row), axis=0)
    unpacked_data = unpacked_data.reshape(-1, 3)

    o255 = numpy.full((len(unpacked_data),1), 255, dtype=numpy.uint8) # 4th channel
    unpacked_data = numpy.append(unpacked_data, o255, axis=1)

    unpacked_data = unpacked_data.reshape(-1, pixels_per_row, 4)

    borderoutput = cv2.copyMakeBorder(
        unpacked_data, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    cv2.imshow("Send", borderoutput)
    if file_index == 0:
        cv2.waitKey(3000)
    if cv2.waitKey(1000) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
