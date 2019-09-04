extern crate proptest;
extern crate wavefront_obj;

use proptest::prelude::*;
use wavefront_obj::obj;

proptest! {
  #[test]
  fn detect_corrupted_data(s in "[:alphanum:]+") {
    let result = obj::parse(s);
    assert!(result.is_err());
  }
}
