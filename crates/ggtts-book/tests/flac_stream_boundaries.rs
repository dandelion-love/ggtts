use ggtts_book::flac::StreamWriter;

#[test]
fn streamed_frames_match_core_for_short_and_split_blocks() {
    for n in [1usize, 15, 16, 17, 4096, 4097, 8193, 23760] {
        let pcm: Vec<i16> = (0..n).map(|i| ((i * 37 % 60000) as i32 - 30000) as i16).collect();
        let path = std::env::temp_dir().join(format!("ggtts-book-boundary-{}-{n}.flac", std::process::id()));
        let mut writer = StreamWriter::create(&path, 24000, n as u64, &[]).unwrap();
        for chunk in pcm.chunks(731) {
            writer.push(chunk).unwrap();
        }
        writer.finish().unwrap();
        let stream = std::fs::read(&path).unwrap();
        let core = ggtts_core::flac::encode_flac(&pcm, 24000).unwrap();
        let frame = |bytes: &[u8]| {
            let mut at = 4;
            loop {
                let last = bytes[at] & 0x80 != 0;
                let len = ((bytes[at + 1] as usize) << 16) | ((bytes[at + 2] as usize) << 8) | bytes[at + 3] as usize;
                at += 4 + len;
                if last {
                    break;
                }
            }
            bytes[at..].to_vec()
        };
        assert_eq!(&stream[8..12], &[0, 16, 16, 0]);
        assert_eq!(frame(&stream), frame(&core), "length {n}");
        let decoded: Vec<i16> = claxon::FlacReader::open(&path)
            .unwrap()
            .samples()
            .map(|x| x.unwrap() as i16)
            .collect();
        assert_eq!(decoded, pcm, "length {n}");
        std::fs::remove_file(path).unwrap();
    }
}
