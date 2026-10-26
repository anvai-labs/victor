def public_helper(path)
  trimmed = path.strip
  wrapped = wrap(trimmed, trimmed.length)
  wrapped
end

def wrap(text, width)
  padded = text.ljust(width)
  padded
end

class SampleService
  def initialize(name)
    @name = name
    @cache = {}
  end

  def process(items)
    results = []
    items.each do |item|
      next if item.empty?
      begin
        results.push(public_helper(item))
      rescue StandardError => e
        results.push(e.to_s)
      end
    end
    results
  end
end
