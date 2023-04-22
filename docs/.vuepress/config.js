module.exports = {
    head: [
      ['link', { rel: 'stylesheet', href: '/styles/custom.css' }],
      ['link', { rel: 'stylesheet', href: 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.css' }],
      ['link', { rel: "stylesheet", href: "https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/2.10.0/github-markdown.min.css"}]
    ],

    markdown: {
      lineNumbers: true,
      extendMarkdown: md => {
        // 禁用 markdown-it-container 插件的 LaTeX 渲染
        // 启用 markdown-it-katex 插件
        md.use(require('markdown-it-katex'));
      }
    },
    title: "Ai LEARNING",
    theme: 'reco',
    themeConfig: {
      type: 'blog',
      // author
      author: 'Computer Scientist',
      //导航
      nav: [
        { text: "首页", link: "/" },
        { text: 'TimeLine', link: '/timeline/', icon: 'reco-date' }
      ],
      // 博客配置
      blogConfig: {
        authorAvatar: '/avatar.jpg',
        category: {
          location: 2, // 在导航栏菜单中所占的位置，默认2
          text: "博客", // 默认文案 “分类”
        },
        tag: {
          location: 3, // 在导航栏菜单中所占的位置，默认3
          text: "Tag", // 默认文案 “标签”
        },
      },
    },
    plugins: [    
      ['vuepress-plugin-code-copy', {
        color: '#39ac37',
        backgroundTransition: 'true'
      }],
    ],
}