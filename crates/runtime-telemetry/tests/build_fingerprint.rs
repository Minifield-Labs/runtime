#[allow(dead_code, clippy::expect_used)]
mod build_script {
    include!("../build.rs");

    struct Scratch(std::path::PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn metal_shader_changes_alter_the_source_fingerprint() {
        let root = Scratch(env::temp_dir().join(format!(
            "minifield-build-fingerprint-{}",
            uuid::Uuid::now_v7()
        )));
        fs::create_dir(&root.0).expect("scratch directory");
        let shader = root.0.join("kernel.metal");
        fs::write(&shader, "kernel void first() {}\n").expect("first shader");
        let mut before = Sha256::new();
        hash_sources(&root.0, &root.0, &mut before).expect("first fingerprint");
        fs::write(&shader, "kernel void second() {}\n").expect("second shader");
        let mut after = Sha256::new();
        hash_sources(&root.0, &root.0, &mut after).expect("second fingerprint");
        assert_ne!(before.finalize(), after.finalize());
    }
}
