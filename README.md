# Image-Watermarking
Watermarking letters into image using maximum length sequence with C++.<br>
It is based watermarked image g(x,y) = f(x,y) + a(x,y) * m(x,y)<br>
where, f(x,y) is original, a(x,y) is defined by HVS, m(x,y) is message.<br>

## Encoding
1. Devide image into 64 * 64 blocks and strech each block into 1 * 4096 array.<br>
2. First 1024 will be synchronization part, and remaining 3072 will be message part.<br>
3. In sync part, m-sequence of 10 degree will be in. 
4. In message part, 12 letters of ASCII code will be in using m-sequence of 8 degree rotated by each ASCII value. 12 letters with 256 will be 3072.
5. Each of m-sequence will be 1 or -1 and it is m(x,y).
6. Calculate variance of each piexl with 5 * 5. It will be HVS, a(x,y).

## Decoding
1. Use Wiener Filter for recover the image.
2. Calculate differance between watermarked image and filtered image.
3. Calculate watermarked message by synchronizing sync part and rotating message part.
