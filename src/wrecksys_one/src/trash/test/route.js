const results = []

export async function GET (request) {
  results.push(1)

  return new Response(JSON.stringify(results), {
    headers: { 'content-type': 'application/json' },
    status: 200
  })
}
