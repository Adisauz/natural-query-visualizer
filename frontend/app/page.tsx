'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:5000'
import ChartComponent from './components/ChartComponent'

interface ChartData {
  title: string
  x_axis: string
  y_axis: string
  chart_type?: string
  data: { label: string; value: number; x?: number; y?: number; [key: string]: any }[]
}

interface Narrative {
  introduction: string
  transitions: string[]
  insights: string[]
  conclusion: string
}

interface ApiResponse {
  success: boolean
  data: ChartData | ChartData[]
  narrative?: Narrative
  question: string
  database?: string
  error?: string
}

interface Database {
  [key: string]: string
}

interface DatabaseResponse {
  success: boolean
  databases: Database
}

export default function Home() {
  const [question, setQuestion] = useState('')
  const [chartData, setChartData] = useState<ChartData | ChartData[] | null>(null)
  const [narrative, setNarrative] = useState<Narrative | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedDatabase, setSelectedDatabase] = useState('chinook')
  const [databases, setDatabases] = useState<Database>({})
  const [loadingDatabases, setLoadingDatabases] = useState(true)
  const [multipleCharts, setMultipleCharts] = useState(true)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!question.trim()) return

    setLoading(true)
    setError(null)
    setChartData(null)
    setNarrative(null)

    try {
      const response = await axios.post<ApiResponse>('http://192.168.0.193:5000/api/ask', {
        question: question.trim(),
        database: selectedDatabase,
        multiple_charts: multipleCharts
      })

      if (response.data.success) {
        setChartData(response.data.data)
        setNarrative(response.data.narrative || null)
      } else {
        setError(response.data.error || 'Unknown error occurred')
      }
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to connect to the backend')
    } finally {
      setLoading(false)
    }
  }

  const clearResults = () => {
    setChartData(null)
    setError(null)
    setQuestion('')
  }

  // Load available databases on component mount
  useEffect(() => {
    const loadDatabases = async () => {
      try {
        const response = await axios.get<DatabaseResponse>('http://192.168.0.193:5000/api/databases')
        if (response.data.success) {
          setDatabases(response.data.databases)
        }
      } catch (err) {
        console.error('Failed to load databases:', err)
      } finally {
        setLoadingDatabases(false)
      }
    }
    
    loadDatabases()
  }, [])

  // Sample questions for each database
  const sampleQuestions = {
    chinook: [
      "Show me top 5 most popular albums with their number of songs",
      "What percentage of sales does each genre represent?",
      "Show me sales trends by year",
      "Which employees have the highest sales?",
      "Show me the correlation between track length and price",
      "List all customer details with their total purchases"
    ],
    world: [
      "Show me the top 10 most populous countries",
      "What percentage of world population does each continent represent?",
      "Show me the correlation between GNP and life expectancy",
      "List all European countries with their detailed statistics",
      "Show me population growth trends by region over time",
      "Which cities in each continent are the largest?"
    ],
    imdb: [
      "Show me the top 10 highest rated movies",
      "What's the distribution of movies by genre?",
      "Show me movie ratings trends by year",
      "Show me the correlation between movie duration and ratings",
      "List detailed information for all sci-fi movies",
      "Which countries produce the most movies?"
    ]
  }

  const currentSamples = sampleQuestions[selectedDatabase as keyof typeof sampleQuestions] || []

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <h1 className="text-xl font-semibold text-white">
          Database Analytics Assistant
        </h1>
        <p className="text-sm text-gray-400 mt-1">
          Ask questions about your databases in natural language
        </p>
      </div>

      {/* Chat Messages Area */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="max-w-4xl mx-auto space-y-4">
          {/* User Question */}
          {question && (
            <div className="flex justify-end">
              <div className="max-w-2xl">
                <div className="bg-blue-600 text-white rounded-lg px-4 py-3 shadow-lg">
                  <p className="text-sm leading-relaxed">{question}</p>
                  <div className="text-xs text-blue-200 mt-2">
                    Database: {selectedDatabase}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* AI Response */}
          {chartData && (
            <div className="flex justify-start">
              <div className="max-w-2xl">
                <div className="bg-gray-800 rounded-lg px-4 py-3 shadow-lg border border-gray-700">
                  <ChartComponent data={chartData} narrative={narrative || undefined} />
                </div>
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="flex justify-start">
              <div className="max-w-2xl">
                <div className="bg-red-900 border border-red-700 rounded-lg px-4 py-3 shadow-lg">
                  <div className="flex items-start space-x-2">
                    <div className="text-red-400 text-lg">⚠️</div>
                    <div className="flex-1">
                      <p className="text-red-400 text-sm font-medium mb-2">Database Error:</p>
                      <div className="bg-red-800 rounded p-3 text-xs font-mono text-red-200 overflow-x-auto">
                        {error}
                      </div>
                      <p className="text-red-400 text-xs mt-2">
                        💡 Try rephrasing your question or asking about different data.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Sample Questions */}
          {!chartData && !error && (
            <div className="flex justify-start">
              <div className="max-w-2xl">
                <div className="bg-gray-800 rounded-lg px-4 py-3 shadow-lg border border-gray-700">
                  <h3 className="text-sm font-medium text-gray-300 mb-3">
                    💡 Try these sample questions:
                  </h3>
                  <div className="grid grid-cols-1 gap-2">
                    {currentSamples.map((sample, index) => (
                      <button
                        key={index}
                        onClick={() => setQuestion(sample)}
                        className="text-left p-3 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm text-gray-300 border border-gray-600 transition-colors"
                        disabled={loading}
                      >
                        {sample}
                      </button>
                    ))}
                  </div>
                  
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="px-6 py-4">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmit}>
            <div className="flex justify-center">
              <div className="w-full max-w-2xl">
                <div className="flex items-center space-x-3">
                  {/* Database Selection */}
                  <div className="flex items-center space-x-2">
                    <label htmlFor="database" className="text-sm font-medium text-gray-300 whitespace-nowrap">
                      Database:
                    </label>
                    {loadingDatabases ? (
                      <div className="flex items-center space-x-2">
                        <div className="w-4 h-4 border-2 border-gray-600 border-t-gray-400 rounded-full animate-spin"></div>
                        <span className="text-gray-400 text-sm">Loading...</span>
                      </div>
                    ) : (
                      <select
                        id="database"
                        value={selectedDatabase}
                        onChange={(e) => setSelectedDatabase(e.target.value)}
                        className="px-3 py-2 border border-gray-600 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-gray-300"
                        disabled={loading}
                      >
                        {Object.entries(databases).map(([key, description]) => (
                          <option key={key} value={key}>
                            {key.charAt(0).toUpperCase() + key.slice(1)}
                          </option>
                        ))}
                      </select>
                    )}
                  </div>

                  {/* Multiple Charts Toggle */}
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="multipleCharts"
                      checked={multipleCharts}
                      onChange={(e) => setMultipleCharts(e.target.checked)}
                      className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500 focus:ring-2"
                      disabled={loading}
                    />
                    <label htmlFor="multipleCharts" className="text-sm text-gray-300 whitespace-nowrap">
                      Multiple Charts
                    </label>
                  </div>

                  {/* Input Field */}
                  <div className="flex-1">
                    <textarea
                      id="question"
                      rows={1}
                      className="w-full px-4 py-3 border border-gray-600 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm bg-gray-700 text-gray-100 placeholder-gray-400"
                      placeholder={
                        selectedDatabase === 'chinook' 
                          ? "Ask about music data, artists, albums, sales..."
                          : selectedDatabase === 'world'
                            ? "Ask about countries, cities, population..."
                            : "Ask about movies, actors, ratings..."
                      }
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      disabled={loading}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleSubmit(e);
                        }
                      }}
                    />
                  </div>

                  {/* Send Button */}
                  <button
                    type="submit"
                    disabled={loading || !question.trim()}
                    className="px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
                  >
                    {loading ? (
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    ) : (
                      'Send'
                    )}
                  </button>
                </div>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
