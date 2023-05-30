import cv2
import numpy
import mss
from tqdm import tqdm


def search_sequence_numpy(arr,seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------    
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------    
    Output : 1D Array of indices in the input array that satisfy the 
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = numpy.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[numpy.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return numpy.where(numpy.convolve(M,numpy.ones((Nseq),dtype=int))>0)[0]
    else:
        return []

pixel_offset = 0

with mss.mss() as sct:
    monitor = {"top": 0, "left": 0, "width": 3840, "height": 2160}
    match_results = []
    start_code = numpy.array([255,127,85,127,255])

    while len(match_results) == 0:
        img = numpy.array(sct.grab(monitor))
        img = numpy.delete(img,numpy.s_[3],axis=2)
        for pixel_offset in range(8):
            imgflat = img.flatten()[pixel_offset:] // 128
            imgflat = numpy.packbits(imgflat, axis=0)
            match_results = search_sequence_numpy(imgflat, start_code)
            if len(match_results) >= 1:
                break
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            exit    
        
    first_match = match_results[0].item()
    print("match found: " + str(first_match))
    bits_per_row = monitor["width"] * 3
    first_match_bit_offset = first_match * 8
    starty = first_match_bit_offset // bits_per_row
    startx = (first_match_bit_offset % bits_per_row) // 3
    startx += (pixel_offset + 2) // 3

    currentX = first_match + len(start_code)

    pixels_per_row = imgflat[currentX].item() + imgflat[currentX+1].item() * 256
    currentX += 2
    total_sequences = imgflat[currentX].item() + imgflat[currentX+1].item() * 256
    currentX += 2
    message_sequence = imgflat[currentX].item() + imgflat[currentX+1].item() * 256
    currentX += 2
    bytes_in_frame = imgflat[currentX].item() + imgflat[currentX+1].item() * 256 + imgflat[currentX+2].item() * 256 * 256
    currentX += 3
    total_header_bytes = currentX
    sequences_recieved = numpy.zeros(total_sequences, dtype=numpy.uint8) 

    header_bytes = (currentX - first_match) // 3
    total_bytes = header_bytes + bytes_in_frame
    total_bits = total_bytes * 8
    totalpixels = (total_bits + 2) // 3 # essentially round up
    totalrows = ((totalpixels - 1) // pixels_per_row) + 1
    monitor = {"top": starty, "left": startx, "width": pixels_per_row, "height": totalrows}

    total_header_bytes = len(start_code) + 9

    with tqdm(total=total_sequences) as pbar:
        pbar.update(numpy.sum([sequences_recieved]).item())
        while(numpy.sum([sequences_recieved]) < total_sequences):
            match_results = []
            while len(match_results) == 0:
                img = numpy.array(sct.grab(monitor))
                img = numpy.delete(img,numpy.s_[3],axis=2)
                imgflat = img.flatten() // 128
                imgflat = numpy.packbits(imgflat, axis=0)
                start_code_flat = start_code.flatten()
                match_results = search_sequence_numpy(imgflat[0:6], start_code_flat)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    exit
                currentX = len(start_code_flat) + 4 # skip pixels_per_row and total_sequences
            message_sequence = imgflat[currentX].item() + imgflat[currentX+1].item() * 256
            if sequences_recieved[message_sequence] == 1:
                continue
            currentX = len(start_code) + 6
            bytes_in_frame = imgflat[currentX].item() + imgflat[currentX+1].item() * 256 + imgflat[currentX+2].item() * 256 * 256
            immutable_bytes = bytes(bytearray(imgflat[total_header_bytes:(total_header_bytes+bytes_in_frame)]))
            with open("download.7z." + str(message_sequence + 1).zfill(3), "wb") as binary_file:
                binary_file.write(immutable_bytes)
            sequences_recieved[message_sequence] = 1    
            pbar.update(1)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    exit
