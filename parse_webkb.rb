require 'set'

def parse_webkb(dir, output_train, output_test, labels_train, labels_test, vocab_file)
  get_dirs = lambda do |dir|
    Dir.entries(dir).select { |f| File.directory? File.join(dir, f) and !(f == '.' || f == '..') }
  end
  categories = get_dirs.call(dir)
  o1 = File.open(output_train, 'w')
  l1 = File.open(labels_train, 'w')
  o2 = File.open(output_test, 'w')
  l2 = File.open(labels_test, 'w')
  corpus = Hash.new(0)
  num_docs = 0
  categories.each_with_index do |group, c|
    group_path = File.join(dir, group)
    unis = get_dirs.call(group_path)
    unis.each_with_index do |uni, i|
      if i < 4
        o = o1
        l = l1
      else
        o = o2
        l = l2
      end
      uni_path = File.join(group_path, uni)
      files = Dir.entries(uni_path).reject { |f| File.directory? File.join(uni_path, f) }
      files.each do |file|
        file_path = File.join(uni_path, file)
        doc = File.read(file_path).scrub('')
        doc = doc.downcase
        doc = doc.gsub('-', ' ')
        doc = doc.gsub(/[^a-z ]/, '')
        doc = doc.gsub(' +', ' ')
        words = doc.split.to_set
        words.each { |word| corpus[word] += 1 }
        label = c
        o.puts(doc)
        l.puts(label)
        num_docs += 1
      end
    end 
  end
  o1.close
  l1.close
  o2.close
  l2.close
  File.open(vocab_file, 'w') do |f|
    corpus.each do |word, count|
      if count >= 5 && count <= num_docs / 4
        f.puts(word)
      end
    end
  end
end

def main
  parse_webkb("webkb", "webkb_docs_train.txt", "webkb_docs_test.txt", "webkb_labels_train.txt", "webkb_labels_test.txt", "webkb_vocabulary.txt")
end

main
