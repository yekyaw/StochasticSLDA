require 'set'

def parse_webkb(dir, output, labels)
  get_dirs = lambda do |dir|
    Dir.entries(dir).select { |f| File.directory? File.join(dir, f) and !(f == '.' || f == '..') }
  end
  categories = get_dirs.call(dir)
  File.open(output, 'w') do |o|
    File.open(labels, 'w') do |l|
      categories.each_with_index do |group, c|
        group_path = File.join(dir, group)
        unis = get_dirs.call(group_path)
        unis.each_with_index do |uni, i|
          if i < 4
            uni_path = File.join(group_path, uni)
            files = Dir.entries(uni_path).reject { |f| File.directory? File.join(uni_path, f) }
            files.each do |file|
              file_path = File.join(uni_path, file)
              doc = File.read(file_path).scrub('').gsub("\n", ' ')
              o.puts(doc)
              label = c
              l.puts(label)
            end
          end
        end
      end
    end
  end
end

def main
  vocab_file = File.read("dictnostops.txt")
  vocab = Set.new(vocab_file.split(/\s+/))
  parse_webkb("webkb", "webkb_docs_train.txt", "webkb_labels_train.txt")
end

main
