//! Verify FLAC timestamps and decoded samples at short, exact and partial block boundaries.
use ggtts_core::flac::encode_flac;
use std::io::Cursor;

fn frame_start_number(bytes: &[u8], at: usize) -> (u64, usize) {
    let first = bytes[at];
    if first < 0x80 {
        return (u64::from(first), 1);
    }
    let n = first.leading_ones() as usize;
    assert!((2..=7).contains(&n));
    let mut value = u64::from(first & (0x7f >> n));
    for &byte in &bytes[at + 1..at + n] {
        assert_eq!(byte & 0xc0, 0x80);
        value = (value << 6) | u64::from(byte & 0x3f);
    }
    (value, n)
}

#[test]
fn variable_blocks_expose_absolute_sample_numbers_and_preserve_pcm() {
    for len in [1usize, 4095, 4096, 4097, 8192, 23760, 32769, 131073] {
        let pcm: Vec<i16> = (0..len).map(|i| (i as i32 * 97 % 65536 - 32768) as i16).collect();
        let flac = encode_flac(&pcm, 24000).unwrap();
        let mut reader = claxon::FlacReader::new(Cursor::new(&flac)).unwrap();
        let info = reader.streaminfo();
        assert_eq!(info.samples, Some(len as u64));
        assert_eq!(info.sample_rate, 24000);
        let decoded: Vec<i32> = reader.samples().collect::<Result<_, _>>().unwrap();
        assert_eq!(
            decoded,
            pcm.iter().map(|&s| i32::from(s)).collect::<Vec<_>>(),
            "len={len}"
        );

        let mut offset = 42;
        let mut sample = 0usize;
        while sample < len {
            assert_eq!(flac[offset], 0xff);
            assert_eq!(flac[offset + 1] & 1, 1, "frame must use variable blocking");
            let (number, number_len) = frame_start_number(&flac, offset + 4);
            assert_eq!(number, sample as u64, "wrong absolute sample at frame");
            let block = (len - sample).min(4096);
            let extra = if block == 4096 { 0 } else { 2 };
            assert_eq!(flac[offset + 2] >> 4, if extra == 0 { 12 } else { 7 });
            // 4-byte prefix, sample number, optional explicit length, header CRC,
            // verbatim subframe byte, PCM bytes, frame CRC.
            offset += 4 + number_len + extra + 1 + 1 + 2 * block + 2;
            sample += block;
        }
        assert_eq!(offset, flac.len(), "frame lengths for {len}");
    }
}
